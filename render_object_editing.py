"""
Render and edit objects based on 3D mask IDs.
Supports:
1. Visualizing all mask IDs in different colors
2. Removing specific objects by mask ID
3. Isolating specific objects
4. Creating object masks
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
import cv2


def mask_id_to_color(mask_id, max_ids=256):
    """Convert mask ID to unique RGB color using HSL color space."""
    if mask_id < 0:
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)  # Gray for unassigned

    # Use golden ratio for better color distribution
    h = (mask_id * 0.618033988749895) % 1.0
    s = 0.7 + (mask_id % 3) * 0.1  # Higher saturation
    l = 0.5 + (mask_id % 4) * 0.08

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return torch.tensor([r, g, b], dtype=torch.float32)


def render_mask_visualization(model_path, iteration, views, gaussians, pipeline, background):
    """Render views with Gaussian mask IDs as colors to show merging."""

    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: Gaussians don't have mask_ids!")
        return

    mask_ids = gaussians.mask_ids
    unique_ids = torch.unique(mask_ids[mask_ids >= 0])
    num_unique = len(unique_ids)

    print(f"\n{'='*80}")
    print(f"3D Mask ID Visualization")
    print(f"{'='*80}")
    print(f"  Total Gaussians: {len(mask_ids):,}")
    print(f"  Assigned Gaussians: {(mask_ids >= 0).sum().item():,}")
    print(f"  Unique 3D mask IDs: {num_unique}")
    print(f"  Average Gaussians per mask: {(mask_ids >= 0).sum().item() / num_unique:.1f}")
    print(f"{'='*80}\n")

    render_path = os.path.join(model_path, f"mask_id_visualization")
    os.makedirs(render_path, exist_ok=True)

    # Generate color map
    colors = torch.zeros(len(mask_ids), 3, device='cuda')
    for gid in range(len(mask_ids)):
        mid = mask_ids[gid].item()
        colors[gid] = mask_id_to_color(mid, num_unique).cuda()

    # Temporarily replace Gaussian colors
    original_features_dc = gaussians._features_dc.clone()
    gaussians._features_dc = colors.unsqueeze(1)  # (N, 1, 3)

    print(f"Rendering {len(views)} views with mask ID colors...")
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        try:
            rendering = render(view, gaussians, pipeline, background)["render"]
            img = rendering.detach().cpu().permute(1, 2, 0).numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))
        except Exception as e:
            print(f"Error rendering view {idx}: {e}")
            continue

    # Restore original colors
    gaussians._features_dc = original_features_dc

    # Create legend with mask IDs
    print(f"\nGenerating legend...")
    legend_size = 60
    legend_cols = min(15, num_unique)
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

        # Add mask ID text
        cv2.putText(legend, f'{int(mid)}',
                   (x_start + 5, y_start + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    Image.fromarray(legend).save(os.path.join(render_path, 'legend.png'))
    print(f"✓ Visualization complete: {render_path}")


def render_object_removal(model_path, iteration, views, gaussians, pipeline, background, remove_ids):
    """Remove specific objects by masking out their Gaussians."""

    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: Gaussians don't have mask_ids!")
        return

    mask_ids = gaussians.mask_ids
    remove_ids = set(remove_ids)

    # Create mask for Gaussians to keep
    keep_mask = torch.ones(len(mask_ids), dtype=torch.bool, device='cuda')
    for rid in remove_ids:
        keep_mask &= (mask_ids != rid)

    num_removed = (~keep_mask).sum().item()
    print(f"\n{'='*80}")
    print(f"Object Removal")
    print(f"{'='*80}")
    print(f"  Removing mask IDs: {sorted(remove_ids)}")
    print(f"  Gaussians removed: {num_removed:,} / {len(mask_ids):,}")
    print(f"  Gaussians remaining: {keep_mask.sum().item():,}")
    print(f"{'='*80}\n")

    render_path = os.path.join(model_path, f"object_removal_{'_'.join(map(str, sorted(remove_ids)))}")
    os.makedirs(render_path, exist_ok=True)

    # Set removed Gaussians to zero opacity
    original_opacity = gaussians._opacity.clone()

    with torch.no_grad():
        gaussians._opacity[~keep_mask] = -1e10  # Very low opacity

    print(f"Rendering {len(views)} views with objects removed...")
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))

    # Restore opacity
    with torch.no_grad():
        gaussians._opacity = original_opacity
    print(f"✓ Object removal complete: {render_path}")


def render_object_isolation(model_path, iteration, views, gaussians, pipeline, background, keep_ids):
    """Keep only specific objects, hide everything else."""

    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: Gaussians don't have mask_ids!")
        return

    mask_ids = gaussians.mask_ids
    keep_ids = set(keep_ids)

    # Create mask for Gaussians to keep
    keep_mask = torch.zeros(len(mask_ids), dtype=torch.bool, device='cuda')
    for kid in keep_ids:
        keep_mask |= (mask_ids == kid)

    num_kept = keep_mask.sum().item()
    print(f"\n{'='*80}")
    print(f"Object Isolation")
    print(f"{'='*80}")
    print(f"  Keeping mask IDs: {sorted(keep_ids)}")
    print(f"  Gaussians kept: {num_kept:,} / {len(mask_ids):,}")
    print(f"{'='*80}\n")

    render_path = os.path.join(model_path, f"object_isolation_{'_'.join(map(str, sorted(keep_ids)))}")
    os.makedirs(render_path, exist_ok=True)

    # Set non-kept Gaussians to zero opacity
    original_opacity = gaussians._opacity.clone()

    with torch.no_grad():
        gaussians._opacity[~keep_mask] = -1e10

    print(f"Rendering {len(views)} views with isolated objects...")
    for idx, view in enumerate(tqdm(views, desc="Rendering")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        img = rendering.detach().cpu().permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(render_path, f'{view.image_name}.png'))

    # Restore opacity
    with torch.no_grad():
        gaussians._opacity = original_opacity
    print(f"✓ Object isolation complete: {render_path}")


def render_object_masks(model_path, iteration, views, gaussians, pipeline, background):
    """Render binary masks for each unique object."""

    if not hasattr(gaussians, 'mask_ids'):
        print("❌ Error: Gaussians don't have mask_ids!")
        return

    mask_ids = gaussians.mask_ids
    unique_ids = torch.unique(mask_ids[mask_ids >= 0])

    print(f"\n{'='*80}")
    print(f"Rendering Object Masks")
    print(f"{'='*80}")
    print(f"  Number of objects: {len(unique_ids)}")
    print(f"{'='*80}\n")

    masks_path = os.path.join(model_path, f"object_masks")
    os.makedirs(masks_path, exist_ok=True)

    original_opacity = gaussians._opacity.clone()

    for obj_id in tqdm(unique_ids.cpu().numpy(), desc="Processing objects"):
        obj_path = os.path.join(masks_path, f"mask_{int(obj_id):03d}")
        os.makedirs(obj_path, exist_ok=True)

        # Keep only this object
        keep_mask = (mask_ids == obj_id)

        with torch.no_grad():
            gaussians._opacity[~keep_mask] = -1e10

        for view in views:
            rendering = render(view, gaussians, pipeline, background)["render"]
            # Convert to grayscale mask
            img = rendering.detach().cpu().permute(1, 2, 0).mean(dim=2).numpy()
            img = (img > 0.01).astype(np.uint8) * 255
            Image.fromarray(img, mode='L').save(os.path.join(obj_path, f'{view.image_name}.png'))

        # Restore opacity
        with torch.no_grad():
            gaussians._opacity = original_opacity.clone()

    print(f"✓ Object masks complete: {masks_path}")


def main():
    parser = ArgumentParser(description="Render and edit objects based on 3D mask IDs")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--mode', type=str, default='visualize',
                       choices=['visualize', 'remove', 'isolate', 'masks', 'all'],
                       help='Operation mode')
    parser.add_argument('--mask_ids', type=int, nargs='+', default=[],
                       help='Mask IDs to remove/isolate')
    args = get_combined_args(parser)

    print(f"\n{'='*80}")
    print("3D Object Editing with Mask IDs")
    print(f"{'='*80}\n")

    # safe_state(args.quiet)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.skip_train:
        views = scene.getTrainCameras()
        if args.mode == 'visualize' or args.mode == 'all':
            render_mask_visualization(args.model_path, scene.loaded_iter, views,
                                     gaussians, pp.extract(args), background)

        if args.mode == 'remove' and args.mask_ids:
            render_object_removal(args.model_path, scene.loaded_iter, views,
                                 gaussians, pp.extract(args), background, args.mask_ids)

        if args.mode == 'isolate' and args.mask_ids:
            render_object_isolation(args.model_path, scene.loaded_iter, views,
                                   gaussians, pp.extract(args), background, args.mask_ids)

        if args.mode == 'masks' or args.mode == 'all':
            render_object_masks(args.model_path, scene.loaded_iter, views,
                               gaussians, pp.extract(args), background)

    print(f"\n{'='*80}")
    print("✓ All operations complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
