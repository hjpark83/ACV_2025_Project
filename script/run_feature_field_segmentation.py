#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentation import FeatureFieldConfig, FeatureFieldPipeline, save_feature_field_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM + LoG + DINO feature field segmentation for static scenes."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory containing RGB images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for mask npz files.")
    parser.add_argument("--sam-checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint.")
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda", help="Device for DINO feature extractor.")
    parser.add_argument("--sam-device", default=None, help="Device for SAM (defaults to --device).")
    parser.add_argument("--sam-fallback-device", default="cpu", help="Fallback device if SAM OOMs on the primary device.")
    parser.add_argument("--dino-model-name", default="dinov2_vits14_reg_lc")
    parser.add_argument("--dino-fallback-models", nargs="*", default=["dinov2_vits14_reg", "dinov2_vits14"], help="Backup model names if the primary DINO cannot load.")
    parser.add_argument("--dino-max-long-edge", type=int, default=1600, help="Long edge cap before feeding DINO.")
    parser.add_argument("--dino-tile-size", type=int, default=0, help="Optional tiling window (pixels) for DINO.")
    parser.add_argument("--dino-tile-stride", type=int, default=None, help="Stride between DINO tiles.")
    parser.add_argument("--dino-cache-dir", type=Path, default=None, help="Directory to cache per-image DINO features.")
    parser.add_argument("--precompute-dino", action="store_true", help="Precompute DINO features into the cache before segmentation.")
    parser.add_argument("--precompute-dino-only", action="store_true", help="Only precompute DINO features and exit.")
    parser.add_argument("--depth-cache-dir", type=Path, default=None, help="Directory with precomputed depth maps (.npy files).")
    parser.add_argument("--depth-method", default="depth_anything_v2", choices=["depth_anything_v2", "midas", "simple"], help="Depth estimation method if not using cache (default: depth_anything_v2).")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth-aware merging.")
    parser.add_argument("--depth-boundary-weight", type=float, default=0.7, help="Penalty weight for depth gradients when scoring merges.")
    parser.add_argument("--depth-gradient-sigma", type=float, default=1.5, help="Gaussian blur sigma before depth gradients.")
    parser.add_argument("--min-mask-area", type=int, default=512)
    # LoG edge detection parameters
    parser.add_argument("--log-sigma", type=float, default=2.0, help="Gaussian blur sigma for LoG edge detection")
    parser.add_argument("--laplacian-ksize", type=int, default=3, help="Laplacian kernel size")

    parser.add_argument("--feature-weight", type=float, default=1.0, help="Weight for feature similarity term.")
    parser.add_argument("--feature-sim-threshold", type=float, default=0.82)
    parser.add_argument("--edge-threshold", type=float, default=0.35)
    parser.add_argument("--edge-penalty", type=float, default=0.5)
    parser.add_argument("--adjacency-dilation", type=int, default=2)
    parser.add_argument("--min-contact-ratio", type=float, default=0.01)
    parser.add_argument("--max-merge-iterations", type=int, default=256)
    parser.add_argument("--feature-margin", type=float, default=0.15, help="Tolerance for minimum feature similarity.")
    parser.add_argument("--adjacency-max-bbox-gap", type=int, default=4, help="Max pixel gap between bounding boxes to treat regions as neighbors.")
    parser.add_argument("--sam-max-long-edge", type=int, default=2048, help="Long edge cap when feeding images to SAM.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images whose outputs already exist.")
    parser.add_argument("--visualize", action="store_true", help="Save visualization overlays for each frame.")
    parser.add_argument("--visualize-dir", type=Path, default=None, help="Directory for visualization outputs (defaults to output-dir/viz).")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image extensions to process.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path, extensions: Sequence[str]) -> List[Path]:
    exts = {ext.lower() for ext in extensions}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def save_result(output_path: Path, image_name: str, result: dict) -> None:
    regions = result["regions"]
    edge_map = result["edge_map"]
    depth_map = result.get("depth_map", None)

    if not regions:
        save_dict = {
            "masks": np.zeros((0, 1, 1), dtype=bool),
            "features": np.zeros((0, 1), dtype=np.float32),
            "areas": np.zeros((0,), dtype=np.int32),
            "bboxes": np.zeros((0, 4), dtype=np.int32),
            "source_ids": np.zeros((0, 1), dtype=np.int32),
            "edge_map": edge_map.astype(np.float32),
            "image_name": image_name,
        }
        if depth_map is not None:
            save_dict["depth_map"] = depth_map.astype(np.float32)
        np.savez_compressed(output_path, **save_dict)
        return

    masks = np.stack([region.mask for region in regions], axis=0).astype(bool)
    features = torch.stack([region.feature for region in regions]).cpu().numpy().astype(np.float32)
    areas = np.array([region.area for region in regions], dtype=np.int32)
    bboxes = np.array([region.bbox for region in regions], dtype=np.int32)
    max_sources = max(len(region.source_ids) for region in regions)
    source_ids = -np.ones((len(regions), max_sources), dtype=np.int32)
    for idx, region in enumerate(regions):
        count = len(region.source_ids)
        source_ids[idx, :count] = region.source_ids

    save_dict = {
        "masks": masks,
        "features": features,
        "areas": areas,
        "bboxes": bboxes,
        "source_ids": source_ids,
        "edge_map": edge_map.astype(np.float32),
        "image_name": image_name,
    }
    if depth_map is not None:
        save_dict["depth_map"] = depth_map.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)


def main() -> None:
    args = parse_args()

    image_dir: Path = args.image_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = None
    if args.visualize:
        viz_dir = args.visualize_dir or (output_dir / "visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
    if args.precompute_dino_only:
        args.precompute_dino = True
    cache_dir: Optional[Path] = args.dino_cache_dir
    if cache_dir is None and (args.precompute_dino or args.precompute_dino_only):
        cache_dir = output_dir / "dino_cache"
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir, args.extensions)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir} with extensions {args.extensions}")

    config = FeatureFieldConfig(
        sam_checkpoint=str(args.sam_checkpoint),
        sam_model_type=args.sam_model_type,
        device=args.device,
        sam_device=args.sam_device,
        sam_fallback_device=args.sam_fallback_device,
        dino_model_name=args.dino_model_name,
        min_mask_area=args.min_mask_area,
        log_sigma=args.log_sigma,
        laplacian_ksize=args.laplacian_ksize,
        feature_weight=args.feature_weight,
        feature_sim_threshold=args.feature_sim_threshold,
        edge_strength_threshold=args.edge_threshold,
        edge_penalty=args.edge_penalty,
        adjacency_dilation=args.adjacency_dilation,
        min_contact_ratio=args.min_contact_ratio,
        max_merge_iterations=args.max_merge_iterations,
        feature_margin=args.feature_margin,
        adjacency_max_bbox_gap=args.adjacency_max_bbox_gap,
        use_depth=not args.no_depth,
        depth_method=args.depth_method,
        depth_cache_dir=str(args.depth_cache_dir) if args.depth_cache_dir else None,
        depth_boundary_weight=args.depth_boundary_weight,
        dino_max_long_edge=args.dino_max_long_edge,
        dino_tile_size=args.dino_tile_size,
        dino_tile_stride=args.dino_tile_stride,
        dino_fallback_models=args.dino_fallback_models,
        dino_cache_dir=str(cache_dir) if cache_dir else None,
        sam_max_long_edge=args.sam_max_long_edge,
        depth_gradient_sigma=args.depth_gradient_sigma,
    )
    pipeline = FeatureFieldPipeline(config)

    if args.precompute_dino:
        print("Precomputing DINO features...")
        for image_path in tqdm(images, desc="Precompute DINO"):
            image = np.array(Image.open(image_path).convert("RGB"))
            pipeline.precompute_dino_feature(image, image_path.stem)
        if args.precompute_dino_only:
            print("DINO precomputation complete. Exiting.")
            return

    for image_path in tqdm(images, desc="Feature-field segmentation"):
        output_path = output_dir / f"{image_path.stem}.npz"
        if args.skip_existing and output_path.exists():
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        result = pipeline.process_image(image, image_name=image_path.stem)
        save_result(output_path, image_path.stem, result)
        if viz_dir is not None:
            save_feature_field_visualizations(
                image=image,
                regions=result["regions"],
                edge_map=result["edge_map"],
                out_dir=viz_dir,
                image_name=image_path.stem,
            )

    print(f"\n✓ Saved refined masks to {output_dir}")


if __name__ == "__main__":
    main()
