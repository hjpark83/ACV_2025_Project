#!/usr/bin/env python3
"""
Foreground Composition Script

Loads a trained Gaussian scene, 가져온 전경 PLY를 변환(이동/스케일/회전)하여
기존 씬에 합성한 뒤 새로운 point cloud를 저장합니다.
"""

import json
import os
from argparse import ArgumentParser
from typing import Sequence

import torch

from arguments import ModelParams, get_combined_args
from scene import Scene, GaussianModel
from grouping import (
    merge_scenes,
    load_subset_from_ply,
)


def any_nonzero(values: Sequence[float], eps: float = 1e-6) -> bool:
    return any(abs(v) > eps for v in values)


def main():
    parser = ArgumentParser(description="Compose foreground Gaussians into target scene")
    model_params = ModelParams(parser, sentinel=True)

    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to load from the target scene")
    parser.add_argument("--foreground_ply", type=str, required=True,
                        help="Path to foreground PLY (exported via test_grouping.py)")
    parser.add_argument("--foreground_sh_degree", type=int, default=3,
                        help="Spherical harmonics degree used when exporting the foreground PLY")

    parser.add_argument("--translation", type=float, nargs=3, metavar=("TX", "TY", "TZ"),
                        default=(0.0, 0.0, 0.0),
                        help="Translation vector applied after rotation/scale")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Uniform scale factor applied about the bounding box center")
    parser.add_argument("--rotation", type=float, nargs=3, metavar=("RX", "RY", "RZ"),
                        default=(0.0, 0.0, 0.0),
                        help="Euler rotation in degrees (ZYX order) around the bounding box center")
    parser.add_argument("--pivot", type=float, nargs=3, default=None,
                        help="Custom pivot for rotation/scale (defaults to foreground bounding box center)")

    parser.add_argument("--output_dir", type=str, default="composed_scene",
                        help="Directory (relative to model path) to store composed results")
    parser.add_argument("--tag", type=str, default="placement",
                        help="Tag appended to the output folder name")
    parser.add_argument("--save_metadata", action="store_true",
                        help="Save JSON metadata describing the composition parameters")

    args = get_combined_args(parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print("Gaussian Composition")
    print(f"{'='*80}\n")

    # Load target scene
    dataset = model_params.extract(args)
    target_gaussians = GaussianModel(dataset.sh_degree)
    Scene(dataset, target_gaussians, load_iteration=args.iteration, shuffle=False)

    print(f"✓ Loaded target scene: {dataset.model_path}")
    print(f"  Iteration: {args.iteration}")
    print(f"  Gaussians: {len(target_gaussians.get_xyz):,}")

    # Load foreground subset from PLY
    subset = load_subset_from_ply(args.foreground_ply,
                                  sh_degree=args.foreground_sh_degree,
                                  device=device if device.type == "cuda" else None)

    print(f"✓ Loaded foreground PLY: {args.foreground_ply}")
    print(f"  Foreground Gaussians: {subset.num_gaussians:,}")

    pivot_tensor = None
    if args.pivot is not None:
        pivot_tensor = torch.tensor(args.pivot, device=subset.device, dtype=subset.dtype)
    else:
        pivot_tensor = subset.center()

    # Apply transformations
    if abs(args.scale - 1.0) > 1e-6:
        subset.apply_scale(args.scale, pivot=pivot_tensor)
        print(f"  Applied scale: {args.scale}")

    if any_nonzero(args.rotation):
        rotation_tensor = torch.tensor(args.rotation, device=subset.device, dtype=subset.dtype)
        subset.apply_rotation(euler=rotation_tensor, pivot=pivot_tensor, degrees=True)
        print(f"  Applied rotation (deg): {args.rotation}")

    if any_nonzero(args.translation):
        translation_tensor = torch.tensor(args.translation, device=subset.device, dtype=subset.dtype)
        subset.apply_translation(translation_tensor)
        print(f"  Applied translation: {args.translation}")

    # Merge and save
    composed_model = merge_scenes(target_gaussians, subset)

    output_root = os.path.join(
        dataset.model_path,
        args.output_dir,
        f"{args.tag}_iter_{args.iteration}"
    )
    os.makedirs(output_root, exist_ok=True)

    ply_path = os.path.join(output_root, "point_cloud.ply")
    composed_model.save_ply(ply_path)
    print(f"\n✓ Saved composed scene to: {ply_path}")

    if args.save_metadata:
        metadata = {
            "target_model_path": dataset.model_path,
            "iteration": args.iteration,
            "foreground_ply": os.path.abspath(args.foreground_ply),
            "foreground_gaussians": subset.num_gaussians,
            "translation": list(args.translation),
            "scale": args.scale,
            "rotation_deg": list(args.rotation),
            "pivot": pivot_tensor.detach().cpu().tolist() if pivot_tensor is not None else None,
            "output_dir": output_root,
        }

        meta_path = os.path.join(output_root, "composition_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to: {meta_path}")

    print(f"\n{'='*80}")
    print("✅ Composition Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
