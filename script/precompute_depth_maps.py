"""
Pre-compute depth maps for all images using Depth Anything V2.

Usage:
    conda activate GG
    python script/precompute_depth_maps.py \
        --image-dir data/lerf/figurines/images \
        --output-dir cache/depth_maps/figurines

    # Or specify method explicitly:
    python script/precompute_depth_maps.py \
        --image-dir data/tree/images \
        --output-dir cache/depth_maps/tree \
        --method depth_anything_v2

"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.depth_estimator import DINOv2DepthEstimator


def main():
    parser = argparse.ArgumentParser(description="Pre-compute depth maps")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save depth maps (.npy files)")
    parser.add_argument("--method", type=str, default="depth_anything_v2", choices=["depth_anything_v2", "midas", "simple"],
                        help="Depth estimation method (default: depth_anything_v2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images that already have depth maps")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Depth Estimation Pre-computation")
    print(f"{'='*60}")
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    estimator = DINOv2DepthEstimator(method=args.method, device=args.device)

    image_dir = Path(args.image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix in image_extensions
    ])

    print(f"Found {len(image_paths)} images\n")

    for image_path in tqdm(image_paths, desc="Computing depth maps"):
        output_path = os.path.join(args.output_dir, f"{image_path.stem}.npy")
        if args.skip_existing and os.path.exists(output_path):
            continue
        image = Image.open(image_path)
        depth_map = estimator.estimate(image)
        np.save(output_path, depth_map.astype(np.float32))

    print(f"\n✓ Depth maps saved to {args.output_dir}")
    print(f"  Total files: {len(image_paths)}")

    if len(image_paths) > 0:
        example_depth = np.load(os.path.join(args.output_dir, f"{image_paths[0].stem}.npy"))
        print(f"  Example depth shape: {example_depth.shape}")
        print(f"  Depth range: [{example_depth.min():.3f}, {example_depth.max():.3f}]")

if __name__ == "__main__":
    main()
