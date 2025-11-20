"""
Render refined masks with ID labels overlaid on boundaries.
This helps visualize which regions have been merged and their global IDs.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import colorsys


def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        sat = 0.6 + (i % 3) * 0.15
        val = 0.7 + (i % 4) * 0.075
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors


def render_masks_with_id_labels(feature_field_dir, output_dir, font_size=12):
    """
    Render refined masks with:
    1. Colored regions (same color = same mask)
    2. Black boundaries
    3. White mask ID labels at region centers
    """

    feature_field_dir = Path(feature_field_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(feature_field_dir.glob("*.npz"))

    if not npz_files:
        print(f"❌ No .npz files found in {feature_field_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Rendering Masks with ID Labels")
    print(f"{'='*80}")
    print(f"  Input: {feature_field_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Files: {len(npz_files)}")
    print(f"{'='*80}\n")

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for npz_path in tqdm(npz_files, desc="Rendering masks"):
        data = np.load(npz_path)

        if 'masks' not in data:
            continue

        masks = data['masks']  # (K, H, W) binary masks
        num_masks = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]

        # Generate distinct colors for this view
        colors = generate_distinct_colors(num_masks)

        # Create colored mask image
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
        mask_id_map = np.zeros((H, W), dtype=np.int32) - 1  # -1 = no mask

        # Fill regions with colors
        for i in range(num_masks):
            mask_i = masks[i] > 0.5
            colored_mask[mask_i] = colors[i]
            mask_id_map[mask_i] = i

        # Find boundaries using morphological operations
        boundary_img = np.zeros((H, W), dtype=np.uint8)
        for i in range(num_masks):
            mask_i = (masks[i] > 0.5).astype(np.uint8) * 255
            # Erode to find inner boundary
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask_i, kernel, iterations=1)
            boundary = mask_i - eroded
            boundary_img = np.maximum(boundary_img, boundary)

        # Draw black boundaries
        colored_mask[boundary_img > 0] = [0, 0, 0]

        # Convert to PIL for text drawing
        pil_img = Image.fromarray(colored_mask)
        draw = ImageDraw.Draw(pil_img)

        # Find region centers and draw ID labels
        for i in range(num_masks):
            mask_i = masks[i] > 0.5

            if not mask_i.any():
                continue

            # Find centroid
            coords = np.argwhere(mask_i)
            if len(coords) == 0:
                continue

            center_y, center_x = coords.mean(axis=0).astype(int)

            # Draw ID label
            label = str(i)

            # Get text size for centering
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Draw white text with black outline for visibility
            outline_width = 2
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text((center_x - text_w//2 + dx, center_y - text_h//2 + dy),
                                 label, font=font, fill=(0, 0, 0))

            draw.text((center_x - text_w//2, center_y - text_h//2),
                     label, font=font, fill=(255, 255, 255))

        # Add mask count annotation
        annotation = f"Masks: {num_masks}"
        draw.text((10, 10), annotation, font=font, fill=(255, 255, 255))
        draw.text((10, 10), annotation, font=font, fill=(0, 0, 0))  # Outline

        # Save
        output_path = output_dir / f"{npz_path.stem}_labeled.png"
        pil_img.save(output_path)

    print(f"\n✓ Rendered {len(npz_files)} masks with ID labels")
    print(f"  Output: {output_dir}")


def create_comparison_video(labeled_dir, output_video_path, fps=30):
    """Create video from labeled mask images."""
    labeled_dir = Path(labeled_dir)
    img_files = sorted(labeled_dir.glob("*_labeled.png"))

    if not img_files:
        print("No labeled images found for video")
        return

    # Read first image to get dimensions
    first_img = cv2.imread(str(img_files[0]))
    H, W = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (W, H))

    print(f"\nCreating video: {output_video_path}")
    for img_path in tqdm(img_files, desc="Writing frames"):
        img = cv2.imread(str(img_path))
        out.write(img)

    out.release()
    print(f"✓ Video saved: {output_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render masks with ID labels")
    parser.add_argument("--feature_field_dir", type=str, required=True,
                       help="Directory containing .npz mask files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: feature_field_dir/labeled_masks)")
    parser.add_argument("--font_size", type=int, default=14,
                       help="Font size for ID labels")
    parser.add_argument("--create_video", action="store_true",
                       help="Create MP4 video from labeled images")
    parser.add_argument("--fps", type=int, default=30,
                       help="Video framerate")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(args.feature_field_dir) / "labeled_masks"

    render_masks_with_id_labels(args.feature_field_dir, args.output_dir, args.font_size)

    if args.create_video:
        video_path = Path(args.output_dir).parent / "masks_labeled.mp4"
        create_comparison_video(args.output_dir, video_path, args.fps)
