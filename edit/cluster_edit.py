"""Cluster-based Gaussian editing utilities."""

import csv
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from typing import List, Optional

import numpy as np
import torch

from arguments import ModelParams, get_combined_args
from scene import Scene, GaussianModel
from utils.general_utils import safe_state


def select_parameters(model: GaussianModel, keep_indices: torch.Tensor) -> None:
    """Filter Gaussian parameters by the provided indices."""

    def reassign(param):
        if isinstance(param, torch.nn.Parameter):
            return torch.nn.Parameter(param[keep_indices].detach().clone())
        return param[keep_indices]

    model._xyz = reassign(model._xyz)
    model._scaling = reassign(model._scaling)
    model._rotation = reassign(model._rotation)
    model._features_dc = reassign(model._features_dc)
    model._features_rest = reassign(model._features_rest)
    model._opacity = reassign(model._opacity)
    model._objects_dc = reassign(model._objects_dc)

    if model._dino_features.numel() > 0:
        model._dino_features = model._dino_features[keep_indices]

    if model.dino_reliability_weights.numel() > 0:
        model.dino_reliability_weights = model.dino_reliability_weights[keep_indices]

    if model.max_radii2D.numel() > 0:
        model.max_radii2D = model.max_radii2D[keep_indices]


def translate_parameters(model: GaussianModel, indices: torch.Tensor, translation: torch.Tensor) -> None:
    """Translate selected Gaussian positions by the given vector."""

    with torch.no_grad():
        model._xyz[indices] = model._xyz[indices] + translation


def load_summary(csv_path: str) -> List[dict]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row["cluster_id"] = int(row["cluster_id"])
                row["total_pixels"] = float(row.get("total_pixels", row.get("pixel_count", 0)))
                row["num_views"] = int(float(row.get("num_views", 1)))
            except Exception:
                continue
            rows.append(row)
    return rows


def parse_args():
    parser = ArgumentParser(description="Cluster-level Gaussian editing")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--labels_path", type=str, default="refined_labels.npy")
    parser.add_argument("--ids", type=int, nargs="+", help="Cluster/global IDs to edit")
    parser.add_argument("--csv_path", type=str, help="Path to cluster summary CSV for suggestions/auto-selection")
    parser.add_argument("--auto_select_topk", type=int, default=0, help="Automatically select top-K clusters by area from CSV")
    parser.add_argument(
        "--operation",
        type=str,
        choices=["remove", "translate"],
        required=True,
        help="Editing operation to apply",
    )
    parser.add_argument(
        "--translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Translation vector (used when operation=translate)",
    )
    parser.add_argument("--output_dir", type=str, default="cluster_edits")
    parser.add_argument("--tag", type=str, default="edit")
    parser.add_argument("--render_after", action="store_true")
    parser.add_argument("--render_split", type=str, choices=["train", "test", "both"], default="train")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, model


def main():
    args, model = parse_args()

    safe_state(args.quiet)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    labels_path = os.path.join(dataset.model_path, args.labels_path)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find cluster labels at {labels_path}")

    cluster_labels = np.load(labels_path)
    if cluster_labels.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(
            f"Label count ({cluster_labels.shape[0]}) does not match Gaussian count ({gaussians.get_xyz.shape[0]})."
        )

    selected_ids: Optional[List[int]] = args.ids
    if (not selected_ids) and args.csv_path:
        rows = load_summary(os.path.join(dataset.model_path, args.csv_path) if not os.path.isabs(args.csv_path) else args.csv_path)
        if not rows:
            raise ValueError("CSV file did not contain any usable cluster rows.")
        rows_sorted = sorted(rows, key=lambda r: r.get("total_pixels", 0), reverse=True)
        if args.auto_select_topk and args.auto_select_topk > 0:
            top = rows_sorted[: min(args.auto_select_topk, len(rows_sorted))]
            selected_ids = [row["cluster_id"] for row in top]
            print("Auto-selected cluster IDs (top by area):", selected_ids)
        else:
            print("Top clusters (by total_pixels):")
            for row in rows_sorted[: min(10, len(rows_sorted))]:
                print(f"  ID {row['cluster_id']}: pixels={row['total_pixels']:.1f}, views={row['num_views']}")
            raise ValueError("No cluster IDs provided. Use --ids or --auto_select_topk with --csv_path.")

    if not selected_ids:
        raise ValueError("Cluster IDs must be specified via --ids or --auto_select_topk/--csv_path.")

    ids = set(int(x) for x in selected_ids)
    mask = np.isin(cluster_labels, list(ids))
    if not mask.any():
        raise ValueError("No Gaussians matched the provided IDs.")

    mask_tensor = torch.from_numpy(mask).to(device="cuda")
    indices = torch.nonzero(mask_tensor, as_tuple=False).squeeze(1)

    if args.operation == "remove":
        keep_indices = torch.nonzero(~mask_tensor, as_tuple=False).squeeze(1)
        select_parameters(gaussians, keep_indices)
        cluster_labels = cluster_labels[~mask]
    elif args.operation == "translate":
        translation = torch.tensor(args.translation, dtype=torch.float32, device="cuda")
        translate_parameters(gaussians, indices, translation)
    else:
        raise NotImplementedError(args.operation)

    output_base = os.path.join(dataset.model_path, args.output_dir, f"{args.operation}_{args.tag}")
    os.makedirs(output_base, exist_ok=True)

    iteration_dir = os.path.join(output_base, "point_cloud", f"iteration_{scene.loaded_iter}")
    os.makedirs(iteration_dir, exist_ok=True)
    ply_path = os.path.join(iteration_dir, "point_cloud.ply")
    gaussians.save_ply(ply_path)
    print(f"Saved edited Gaussian point cloud to {ply_path}")

    classifier_src = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth")
    if os.path.exists(classifier_src):
        shutil.copy2(classifier_src, os.path.join(iteration_dir, "classifier.pth"))

    cfg_src = os.path.join(dataset.model_path, "cfg_args")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(output_base, "cfg_args"))

    if args.operation == "remove":
        labels_output = os.path.join(output_base, "refined_labels.npy")
        np.save(labels_output, cluster_labels)
        print(f"Saved updated labels to {labels_output}")

    if args.render_after:
        render_script = os.path.join(os.path.dirname(__file__), "render.py")
        render_cmd = [
            sys.executable,
            render_script,
            "-m",
            output_base,
            "--iteration",
            str(scene.loaded_iter),
            "--quiet",
        ]
        if args.render_split == "train":
            render_cmd.append("--skip_test")
        elif args.render_split == "test":
            render_cmd.append("--skip_train")
        print("Rendering edited result:", " ".join(render_cmd))
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.run(render_cmd, check=True, env=env)


if __name__ == "__main__":
    main()
