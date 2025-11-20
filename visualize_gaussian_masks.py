"""
Visualize Gaussian 3D point cloud with mask ID colors.
Requires: pip install open3d
"""

import torch
import numpy as np
import open3d as o3d
from scene import Scene, GaussianModel
from arguments import ModelParams
from argparse import ArgumentParser
import colorsys


def mask_id_to_color(mask_id, max_ids=256):
    """Convert mask ID to unique RGB color."""
    if mask_id < 0:
        return np.array([0.5, 0.5, 0.5])  # Gray for unassigned

    h = (mask_id * 0.618033988749895) % 1.0
    s = 0.6 + (mask_id % 3) * 0.1
    l = 0.5 + (mask_id % 4) * 0.08

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([r, g, b])


def visualize_gaussian_point_cloud(model_path, iteration):
    """Visualize Gaussians as 3D point cloud colored by mask ID."""

    print(f"Loading model from: {model_path}")

    # Load config from saved model
    import os
    import re
    cfg_args_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_args_path):
        print(f"❌ Error: Config file not found at {cfg_args_path}")
        return

    with open(cfg_args_path, 'r') as f:
        args_text = f.read()

    # Parse Namespace string
    def parse_value(key, default):
        match = re.search(rf"{key}='?([^',]*)'?", args_text)
        if match:
            val = match.group(1).strip()
            # Try to convert to appropriate type
            if val == 'True':
                return True
            elif val == 'False':
                return False
            elif val.lstrip('-').isdigit():  # Handle negative numbers
                return int(val)
            try:
                return float(val)
            except ValueError:
                return val
        return default

    class Args:
        pass

    args = Args()
    args.model_path = model_path
    args.sh_degree = 3
    args.source_path = parse_value('source_path', '')
    args.images = parse_value('images', 'images')
    args.resolution = parse_value('resolution', -1)
    args.white_background = parse_value('white_background', False)
    args.data_device = parse_value('data_device', 'cuda')
    args.eval = parse_value('eval', False)
    args.object_path = parse_value('object_path', 'object_mask')
    args.n_views = parse_value('n_views', 100)
    args.random_init = parse_value('random_init', False)
    args.train_split = parse_value('train_split', False)

    print(f"Source path: {args.source_path}")

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Get Gaussian data
    xyz = gaussians.get_xyz.detach().cpu().numpy()  # (N, 3)

    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: No mask_ids found in model!")
        print("   This model was trained without 3D mask assignment.")
        return

    mask_ids = gaussians.mask_ids.cpu().numpy()  # (N,)

    print(f"\n{'='*80}")
    print(f"Gaussian Point Cloud Statistics")
    print(f"{'='*80}")
    print(f"  Total Gaussians: {len(xyz)}")
    print(f"  Assigned Gaussians: {(mask_ids >= 0).sum()}")
    print(f"  Unique mask IDs: {len(np.unique(mask_ids[mask_ids >= 0]))}")
    print(f"{'='*80}\n")

    # Generate colors for each Gaussian based on mask ID
    colors = np.zeros((len(mask_ids), 3))
    unique_ids = np.unique(mask_ids[mask_ids >= 0])

    print(f"Generating colors for {len(unique_ids)} unique masks...")
    for i, gid in enumerate(range(len(mask_ids))):
        mid = mask_ids[gid]
        colors[gid] = mask_id_to_color(int(mid), len(unique_ids))

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Print mask ID statistics
    print("\nMask ID Distribution:")
    unique, counts = np.unique(mask_ids[mask_ids >= 0], return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    print(f"  Top 10 largest masks:")
    for i in sorted_indices[:10]:
        mid = unique[i]
        count = counts[i]
        print(f"    Mask {mid:3d}: {count:6d} Gaussians ({count/len(xyz)*100:.1f}%)")

    # Visualize
    print(f"\nLaunching Open3D visualizer...")
    print(f"  - Rotate: Left mouse drag")
    print(f"  - Zoom: Scroll wheel")
    print(f"  - Pan: Right mouse drag")
    print(f"  - Press 'Q' to exit")

    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"3D Gaussian Masks - Iteration {iteration}",
                                      width=1280, height=720,
                                      point_show_normal=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize Gaussian 3D masks")
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--iteration", default=50000, type=int)
    args = parser.parse_args()

    visualize_gaussian_point_cloud(args.model_path, args.iteration)
