"""
Render 3D Mask IDs as colors to visualize merging results.
"""

import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import colorsys


def mask_id_to_color(mask_id, max_ids=256):
    """Convert mask ID to unique RGB color using HSL color space."""
    if mask_id < 0:
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)  # Gray for unassigned

    # Use golden ratio for better color distribution
    h = (mask_id * 0.618033988749895) % 1.0
    s = 0.6 + (mask_id % 3) * 0.1
    l = 0.5 + (mask_id % 4) * 0.08

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return torch.tensor([r, g, b], dtype=torch.float32)


def render_mask_ids(model_path, iteration, views, gaussians, pipeline, background):
    """Render views with Gaussian mask IDs as colors."""

    # Check if mask_ids exist
    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: Gaussians don't have mask_ids attribute!")
        print("   This model was trained without 3D mask assignment.")
        return

    mask_ids = gaussians.mask_ids
    print(f"\n{'='*80}")
    print(f"Rendering 3D Mask IDs")
    print(f"{'='*80}")
    print(f"  Total Gaussians: {len(mask_ids)}")
    print(f"  Assigned Gaussians: {(mask_ids >= 0).sum().item()}")
    print(f"  Unique mask IDs: {len(torch.unique(mask_ids[mask_ids >= 0]))}")
    print(f"{'='*80}\n")

    # Create output directory
    render_path = os.path.join(model_path, f"mask_id_renders_{iteration}")
    os.makedirs(render_path, exist_ok=True)

    # Generate color map for each Gaussian
    unique_ids = torch.unique(mask_ids[mask_ids >= 0])
    num_unique = len(unique_ids)
    print(f"Generating colors for {num_unique} unique mask IDs...")

    # Create color tensor for all Gaussians
    colors = torch.zeros(len(mask_ids), 3, device='cuda')
    for i, gid in enumerate(tqdm(range(len(mask_ids)), desc="Assigning colors")):
        mid = mask_ids[gid].item()
        colors[gid] = mask_id_to_color(mid, num_unique).cuda()

    print(f"\nRendering {len(views)} views...")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Render with mask ID colors
        rendering = render(view, gaussians, pipeline, background, override_color=colors)["render"]

        # Save image
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))

    print(f"\n✓ Mask ID rendering complete!")
    print(f"  Output: {render_path}")

    # Generate legend
    print(f"\nGenerating mask ID legend...")
    legend_size = 50
    legend_cols = min(10, num_unique)
    legend_rows = (num_unique + legend_cols - 1) // legend_cols
    legend = np.ones((legend_rows * legend_size, legend_cols * legend_size, 3), dtype=np.uint8) * 255

    for i, mid in enumerate(unique_ids.cpu().numpy()):
        row = i // legend_cols
        col = i % legend_cols
        color = mask_id_to_color(int(mid), num_unique).numpy()
        y_start = row * legend_size
        y_end = (row + 1) * legend_size
        x_start = col * legend_size
        x_end = (col + 1) * legend_size
        legend[y_start:y_end, x_start:x_end] = (color * 255).astype(np.uint8)

    Image.fromarray(legend).save(os.path.join(render_path, 'legend.png'))
    print(f"✓ Legend saved: {os.path.join(render_path, 'legend.png')}")


def main():
    parser = ArgumentParser(description="Render 3D mask IDs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering 3D Mask IDs")

    safe_state(args.quiet)

    # Load model
    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if model.extract(args).white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render train and test views
    if not args.skip_train:
        render_mask_ids(model.extract(args).model_path, scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline.extract(args), background)

    if not args.skip_test:
        render_mask_ids(model.extract(args).model_path, scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline.extract(args), background)


if __name__ == "__main__":
    main()
