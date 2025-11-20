# Copyright (C) 2023, Gaussian-Grouping
# Modified for Feature-Field visualization

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA


def id2rgb(id, max_num_obj=256):
    """Convert region ID to unique RGB color"""
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    if id == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Use HSL color space for better color distribution
    h = (id * 0.618033988749895) % 1.0  # Golden ratio
    s = 0.5 + (id % 3) * 0.15
    l = 0.4 + (id % 4) * 0.1

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)


def visualize_regions(region_map):
    """
    Visualize region map with colors.
    region_map: (H, W) array of region IDs
    Returns: (H, W, 3) RGB image
    """
    H, W = region_map.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)

    unique_ids = np.unique(region_map)
    for region_id in unique_ids:
        rgb_mask[region_map == region_id] = id2rgb(int(region_id))

    return rgb_mask


def create_mask_comparison(initial_mask, refined_mask, num_initial, num_refined):
    """
    Create side-by-side comparison of initial and refined masks.

    Args:
        initial_mask: (H, W, 3) RGB image of initial SAM masks
        refined_mask: (H, W, 3) RGB image of refined masks
        num_initial: Number of initial regions
        num_refined: Number of refined regions

    Returns:
        (H, W*2 + divider, 3) RGB image with side-by-side comparison
    """
    H, W = initial_mask.shape[:2]

    # Create divider line (white, 4 pixels wide)
    divider_width = 4
    divider = np.ones((H, divider_width, 3), dtype=np.uint8) * 255

    # Add text labels
    initial_with_text = initial_mask.copy()
    refined_with_text = refined_mask.copy()

    # Add text to images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Initial mask text
    text_initial = f"Initial SAM: {num_initial} regions"
    text_size = cv2.getTextSize(text_initial, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30

    # Add black background for text
    cv2.rectangle(initial_with_text,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    cv2.putText(initial_with_text, text_initial, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)

    # Refined mask text
    text_refined = f"Refined: {num_refined} regions"
    reduction = num_initial - num_refined
    reduction_pct = (reduction / num_initial * 100) if num_initial > 0 else 0
    text_refined2 = f"({reduction} merged, -{reduction_pct:.1f}%)"

    text_size = cv2.getTextSize(text_refined, font, font_scale, thickness)[0]
    cv2.rectangle(refined_with_text,
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 0, 0), -1)
    cv2.putText(refined_with_text, text_refined, (text_x, text_y),
                font, font_scale, (0, 255, 0), thickness)

    # Add second line of text
    text_size2 = cv2.getTextSize(text_refined2, font, font_scale * 0.7, thickness - 1)[0]
    text_y2 = text_y + 30
    cv2.rectangle(refined_with_text,
                  (text_x - 5, text_y2 - text_size2[1] - 5),
                  (text_x + text_size2[0] + 5, text_y2 + 5),
                  (0, 0, 0), -1)
    cv2.putText(refined_with_text, text_refined2, (text_x, text_y2),
                font, font_scale * 0.7, (0, 255, 0), thickness - 1)

    # Concatenate: Initial | Divider | Refined
    comparison = np.hstack([initial_with_text, divider, refined_with_text])

    return comparison


def load_feature_field_data(feature_field_dir, image_name):
    """
    Load all data from feature-field .npz file.
    Returns dict with: refined_masks, initial_masks, edge_map, dino_features, features_viz
    """
    base_name = os.path.splitext(image_name)[0]

    # Try to find matching file
    possible_names = [
        f"{base_name}.npz",
        f"frame_{base_name}.npz",
    ]

    # Try numeric conversion
    if base_name.isdigit():
        possible_names.append(f"frame_{int(base_name):05d}.npz")

    mask_path = None
    for name in possible_names:
        test_path = os.path.join(feature_field_dir, name)
        if os.path.exists(test_path):
            mask_path = test_path
            break

    if mask_path is None:
        return None

    try:
        data = np.load(mask_path, allow_pickle=True)

        result = {}

        # 1. Refined masks (after hierarchical merging)
        if 'masks' in data:
            masks = data['masks']  # (N_refined, H, W) boolean
            N, H, W = masks.shape

            # Create region ID map
            region_map = np.zeros((H, W), dtype=np.int32)
            for i in range(N):
                region_map[masks[i]] = i + 1

            result['refined_masks'] = visualize_regions(region_map)
            result['num_refined_regions'] = N

        # 2. Initial SAM masks (from source_ids)
        if 'source_ids' in data and 'masks' in data:
            source_ids = data['source_ids']  # (N_refined, max_sources)
            masks = data['masks']

            # Reconstruct initial SAM regions by assigning each source_id
            max_initial_id = source_ids.max()
            initial_map = np.zeros((H, W), dtype=np.int32)

            # Create mapping: for each refined region, assign all its source regions
            for refined_idx in range(len(masks)):
                # Get source IDs for this refined region
                sources = source_ids[refined_idx]
                sources = sources[sources >= 0]  # Filter out -1 padding

                # For visualization: assign each source a unique ID
                for src_id in sources:
                    if src_id >= 0:
                        # Assign this source ID to pixels in refined mask
                        initial_map[masks[refined_idx]] = src_id

            result['initial_masks'] = visualize_regions(initial_map)
            result['num_initial_regions'] = int(max_initial_id) + 1

        # 5. Depth map visualization
        if 'depth_map' in data:
            depth_map = data['depth_map']  # (H, W) float

            # Normalize to [0, 255]
            depth_norm = depth_map.copy()
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
            depth_viz = (depth_norm * 255).astype(np.uint8)

            # Apply colormap for better visualization
            result['depth_map'] = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)

        # 3. Edge map (LoG)
        if 'edge_map' in data:
            edge_map = data['edge_map']  # (H, W) float

            # Normalize to [0, 255]
            edge_map_norm = edge_map.copy()
            edge_map_norm = (edge_map_norm - edge_map_norm.min()) / (edge_map_norm.max() - edge_map_norm.min() + 1e-8)
            edge_map_viz = (edge_map_norm * 255).astype(np.uint8)

            # Convert to 3-channel for visualization
            result['edge_map'] = cv2.applyColorMap(edge_map_viz, cv2.COLORMAP_JET)

        # 4. DINO features visualization (PCA per region)
        if 'features' in data and 'masks' in data:
            features = data['features']  # (N, 384)
            masks = data['masks']

            if features.shape[0] > 0:
                # Apply PCA to reduce 384D -> 3D
                pca = PCA(n_components=3)
                features_reduced = pca.fit_transform(features)  # (N, 3)

                # Normalize to [0, 1]
                features_reduced = (features_reduced - features_reduced.min(axis=0)) / \
                                  (features_reduced.max(axis=0) - features_reduced.min(axis=0) + 1e-8)

                # Create feature visualization
                dino_viz = np.zeros((H, W, 3), dtype=np.float32)
                for i in range(len(masks)):
                    dino_viz[masks[i]] = features_reduced[i]

                result['dino_features'] = (dino_viz * 255).astype(np.uint8)

        return result

    except Exception as e:
        print(f"Warning: Failed to load feature-field data for {image_name}: {e}")
        return None


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, feature_field_dir=None):
    """
    Render multiple views with feature-field visualizations.

    Outputs:
    - renders: RGB rendering
    - gt: Ground truth
    - refined_masks: SAM masks after hierarchical merging
    - initial_masks: Initial SAM masks before merging
    - mask_comparison: Side-by-side comparison (Initial | Refined)
    - edge_maps: LoG edge detection
    - dino_features: DINO feature PCA visualization
    - depth_maps: Depth map visualization
    """
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    refined_mask_path = os.path.join(model_path, name, f"ours_{iteration}", "refined_masks")
    initial_mask_path = os.path.join(model_path, name, f"ours_{iteration}", "initial_sam_masks")
    comparison_path = os.path.join(model_path, name, f"ours_{iteration}", "mask_comparison")
    edge_map_path = os.path.join(model_path, name, f"ours_{iteration}", "edge_maps")
    dino_feature_path = os.path.join(model_path, name, f"ours_{iteration}", "dino_features")
    depth_map_path = os.path.join(model_path, name, f"ours_{iteration}", "depth_maps")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(refined_mask_path, exist_ok=True)
    makedirs(initial_mask_path, exist_ok=True)
    makedirs(comparison_path, exist_ok=True)
    makedirs(edge_map_path, exist_ok=True)
    makedirs(dino_feature_path, exist_ok=True)
    makedirs(depth_map_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Rendering {name} set with {len(views)} views")
    print(f"Feature-field dir: {feature_field_dir if feature_field_dir else 'Not available'}")
    print(f"{'='*80}\n")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 1. Render RGB
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]

        # Save RGB and GT
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))

        # 2. Load and save feature-field visualizations
        if feature_field_dir is not None:
            ff_data = load_feature_field_data(feature_field_dir, view.image_name)

            if ff_data is not None:
                initial_mask_img = None
                refined_mask_img = None

                if 'refined_masks' in ff_data:
                    refined_mask_img = ff_data['refined_masks']
                    Image.fromarray(refined_mask_img).save(
                        os.path.join(refined_mask_path, f'{idx:05d}.png'))

                if 'initial_masks' in ff_data:
                    initial_mask_img = ff_data['initial_masks']
                    Image.fromarray(initial_mask_img).save(
                        os.path.join(initial_mask_path, f'{idx:05d}.png'))

                # Create side-by-side comparison: Initial | Refined
                if initial_mask_img is not None and refined_mask_img is not None:
                    comparison_img = create_mask_comparison(
                        initial_mask_img, refined_mask_img,
                        ff_data.get('num_initial_regions', 0),
                        ff_data.get('num_refined_regions', 0)
                    )
                    Image.fromarray(comparison_img).save(
                        os.path.join(comparison_path, f'{idx:05d}.png'))

                if 'edge_map' in ff_data:
                    Image.fromarray(ff_data['edge_map']).save(
                        os.path.join(edge_map_path, f'{idx:05d}.png'))

                if 'dino_features' in ff_data:
                    Image.fromarray(ff_data['dino_features']).save(
                        os.path.join(dino_feature_path, f'{idx:05d}.png'))

                if 'depth_map' in ff_data:
                    Image.fromarray(ff_data['depth_map']).save(
                        os.path.join(depth_map_path, f'{idx:05d}.png'))

    # Create concatenated video
    create_concat_video(model_path, name, iteration, feature_field_dir is not None)

    print(f"\n✓ Rendering complete!")
    print(f"  Results saved to: {os.path.join(model_path, name, f'ours_{iteration}')}")


def create_concat_video(model_path, name, iteration, has_feature_field):
    """Create concatenated visualization video"""
    base_path = os.path.join(model_path, name, f"ours_{iteration}")

    gts_path = os.path.join(base_path, "gt")
    render_path = os.path.join(base_path, "renders")
    refined_mask_path = os.path.join(base_path, "refined_masks")
    initial_mask_path = os.path.join(base_path, "initial_sam_masks")
    comparison_path = os.path.join(base_path, "mask_comparison")
    edge_map_path = os.path.join(base_path, "edge_maps")
    dino_feature_path = os.path.join(base_path, "dino_features")
    depth_map_path = os.path.join(base_path, "depth_maps")

    out_path = os.path.join(base_path, 'concat')
    makedirs(out_path, exist_ok=True)

    # Check which visualizations exist
    paths_to_concat = [
        ("GT", gts_path),
        ("Render", render_path),
    ]

    if has_feature_field:
        # Add mask comparison (side-by-side) if available
        if os.path.exists(comparison_path) and len(os.listdir(comparison_path)) > 0:
            paths_to_concat.append(("Mask Comparison", comparison_path))
        if os.path.exists(depth_map_path) and len(os.listdir(depth_map_path)) > 0:
            paths_to_concat.append(("Depth Map", depth_map_path))
        if os.path.exists(edge_map_path) and len(os.listdir(edge_map_path)) > 0:
            paths_to_concat.append(("LoG Edges", edge_map_path))
        if os.path.exists(dino_feature_path) and len(os.listdir(dino_feature_path)) > 0:
            paths_to_concat.append(("DINO Features", dino_feature_path))

    print(f"\nCreating concatenated video with {len(paths_to_concat)} columns:")
    for label, _ in paths_to_concat:
        print(f"  - {label}")

    # Setup video writer - need to calculate total width dynamically
    # because mask_comparison is 2x width
    first_img = np.array(Image.open(os.path.join(gts_path, sorted(os.listdir(gts_path))[0])))
    H, W = first_img.shape[:2]

    # Calculate total width based on actual image sizes
    file_names = sorted(os.listdir(gts_path))
    if len(file_names) == 0:
        print("No files to concat!")
        return

    # Load first frame to calculate total width
    first_images = []
    for label, path in paths_to_concat:
        img_path = os.path.join(path, file_names[0])
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            first_images.append(img)
        else:
            first_images.append(np.zeros((H, W, 3), dtype=np.uint8))

    # Get actual concat width
    first_concat = np.hstack(first_images)
    total_width = first_concat.shape[1]
    total_height = first_concat.shape[0]

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps = 5.0 if 'train' in name else 1.0
    writer = cv2.VideoWriter(
        os.path.join(out_path, 'result.mp4'),
        fourcc, fps, (total_width, total_height)
    )

    # Concatenate frames
    for file_name in tqdm(file_names, desc="Creating video"):
        images = []

        for label, path in paths_to_concat:
            img_path = os.path.join(path, file_name)
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path))
                # Ensure 3 channels
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                images.append(img)
            else:
                # Create placeholder with same dimensions as first frame
                # Use the shape from first_images at the same index
                if len(first_images) > len(images):
                    placeholder_shape = first_images[len(images)].shape
                    images.append(np.zeros(placeholder_shape, dtype=np.uint8))
                else:
                    images.append(np.zeros((H, W, 3), dtype=np.uint8))

        # Concatenate horizontally
        concat_img = np.hstack(images)

        # Save image
        Image.fromarray(concat_img).save(os.path.join(out_path, file_name))

        # Write to video
        writer.write(concat_img[:, :, ::-1])  # RGB to BGR

    writer.release()
    print(f"✓ Video saved: {os.path.join(out_path, 'result.mp4')}")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get feature-field directory
        feature_field_dir = getattr(dataset, 'feature_field_dir', None)
        if feature_field_dir and os.path.exists(feature_field_dir):
            print(f"✓ Using feature-field masks from: {feature_field_dir}")
        else:
            print("⚠ No feature-field directory found")
            feature_field_dir = None

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                      gaussians, pipeline, background, feature_field_dir)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                      gaussians, pipeline, background, feature_field_dir)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script with feature-field visualization")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp.extract(args), args.iteration, pp.extract(args), args.skip_train, args.skip_test)
