import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree


def compute_screen_coverage(
    gaussians,
    cameras,
    threshold_pixels: float = 10.0
) -> torch.Tensor:
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()

    coverage_scores = np.zeros(len(xyz))

    for cam_idx, camera in enumerate(cameras[:20]): 

        R = camera.R if isinstance(camera.R, np.ndarray) else camera.R.cpu().numpy()
        T = camera.T if isinstance(camera.T, np.ndarray) else camera.T.cpu().numpy()

        # Transform to camera space
        xyz_cam = np.matmul(xyz, R.T) + T
        depths = xyz_cam[:, 2]

        # Visible Gaussians
        visible = depths > 0.01

        # Approximate pixel coverage (scale / depth)
        scale_max = scales.max(axis=1)  # Max scale
        pixel_coverage = (scale_max / np.maximum(depths, 0.01)) * camera.image_width

        coverage_scores += visible * pixel_coverage

    return torch.from_numpy(coverage_scores).float()


def sparsify_gaussians(
    gaussians,
    cameras,
    opacity_threshold: float = 0.005,
    scale_threshold: float = 0.1,
    coverage_threshold: float = 5.0,
    redundancy_threshold: float = 0.02,
    target_reduction: float = 0.5,
    device: str = "cuda"
) -> Tuple[torch.Tensor, dict]:
    
    num_gaussians = len(gaussians.get_xyz)

    print(f"\nSparsifying {num_gaussians:,} Gaussians...")

    # Get Gaussian properties
    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach()
    scales = gaussians.get_scaling.detach()

    # Initialize keep mask
    keep_mask = torch.ones(num_gaussians, dtype=torch.bool, device=device)

    stats = {
        "original": num_gaussians,
        "removed_opacity": 0,
        "removed_scale": 0,
        "removed_coverage": 0,
        "removed_redundancy": 0
    }

    # 1. Remove low opacity Gaussians
    opacity_mask = opacity.squeeze() > opacity_threshold
    removed = (~opacity_mask).sum().item()
    keep_mask &= opacity_mask
    stats["removed_opacity"] = removed
    print(f"  [1/4] Opacity pruning: removed {removed:,} Gaussians (α < {opacity_threshold})")

    # 2. Remove too large Gaussians (outliers)
    scale_max = scales.max(dim=1)[0]
    scale_mask = scale_max < scale_threshold
    removed = (~scale_mask).sum().item()
    keep_mask &= scale_mask
    stats["removed_scale"] = removed
    print(f"  [2/4] Scale pruning: removed {removed:,} Gaussians (scale > {scale_threshold})")

    # 3. Remove low coverage Gaussians
    print(f"  [3/4] Computing screen coverage...")
    coverage_scores = compute_screen_coverage(gaussians, cameras, threshold_pixels=5.0)
    coverage_scores = coverage_scores.to(device)

    coverage_mask = coverage_scores > coverage_threshold
    removed = (~coverage_mask).sum().item()
    keep_mask &= coverage_mask
    stats["removed_coverage"] = removed
    print(f"  [3/4] Coverage pruning: removed {removed:,} Gaussians (coverage < {coverage_threshold})")

    # 4. Remove redundant Gaussians (optional, expensive)
    if redundancy_threshold > 0:
        print(f"  [4/4] Finding redundant Gaussians...")

        # Only check among remaining Gaussians
        remaining_indices = torch.where(keep_mask)[0]
        xyz_remaining = xyz[remaining_indices].cpu().numpy()

        if len(xyz_remaining) > 10000:
            # For efficiency, only check redundancy for subset
            print(f"       (Sampling due to large size: {len(xyz_remaining):,} Gaussians)")
            sample_size = 10000
            sample_indices = np.random.choice(len(xyz_remaining), sample_size, replace=False)
            xyz_sample = xyz_remaining[sample_indices]

            # Build KD-tree
            tree = cKDTree(xyz_sample)

            # Find close pairs
            close_pairs = tree.query_pairs(redundancy_threshold)

            # Mark redundant ones (keep lower opacity)
            redundant_local = set()
            opacity_remaining = opacity[remaining_indices].cpu().numpy()

            for i, j in close_pairs:
                # Keep the one with higher opacity
                if opacity_remaining[sample_indices[i]] < opacity_remaining[sample_indices[j]]:
                    redundant_local.add(sample_indices[i])
                else:
                    redundant_local.add(sample_indices[j])

            # Map back to global indices
            redundant_global = remaining_indices[list(redundant_local)]
            keep_mask[redundant_global] = False

            removed = len(redundant_global)
            stats["removed_redundancy"] = removed
            print(f"  [4/4] Redundancy pruning: removed {removed:,} Gaussians (distance < {redundancy_threshold})")
        else:
            print(f"       (Skipped due to small size: {len(xyz_remaining):,} Gaussians)")
            stats["removed_redundancy"] = 0
    else:
        print(f"  [4/4] Redundancy pruning: skipped")
        stats["removed_redundancy"] = 0

    # Final statistics
    num_kept = keep_mask.sum().item()
    num_removed = num_gaussians - num_kept
    reduction_ratio = num_removed / num_gaussians

    stats["kept"] = num_kept
    stats["removed_total"] = num_removed
    stats["reduction_ratio"] = reduction_ratio

    print(f"\n  Summary:")
    print(f"    Original: {num_gaussians:,} Gaussians")
    print(f"    Kept: {num_kept:,} Gaussians")
    print(f"    Removed: {num_removed:,} Gaussians ({reduction_ratio*100:.1f}%)")

    if reduction_ratio < target_reduction:
        print(f"\n  ⚠ Warning: Reduction {reduction_ratio*100:.1f}% < target {target_reduction*100:.1f}%")
        print(f"    Consider lowering thresholds for more aggressive pruning.")

    return keep_mask, stats


def apply_sparsification(gaussians, keep_mask):
    
    from scene import GaussianModel

    sparse_gaussians = GaussianModel(gaussians.max_sh_degree)

    # Filter all attributes
    sparse_gaussians._xyz = gaussians._xyz[keep_mask]
    sparse_gaussians._features_dc = gaussians._features_dc[keep_mask]
    sparse_gaussians._features_rest = gaussians._features_rest[keep_mask]
    sparse_gaussians._scaling = gaussians._scaling[keep_mask]
    sparse_gaussians._rotation = gaussians._rotation[keep_mask]
    sparse_gaussians._opacity = gaussians._opacity[keep_mask]

    # Optional attributes
    if hasattr(gaussians, '_dino_features') and gaussians._dino_features is not None:
        sparse_gaussians._dino_features = gaussians._dino_features[keep_mask]

    if hasattr(gaussians, '_objects_dc') and gaussians._objects_dc is not None:
        sparse_gaussians._objects_dc = gaussians._objects_dc[keep_mask]

    if hasattr(gaussians, '_semantic_ids') and gaussians._semantic_ids is not None:
        sparse_gaussians._semantic_ids = gaussians._semantic_ids[keep_mask]

    if hasattr(gaussians, '_semantic_confidence') and gaussians._semantic_confidence is not None:
        sparse_gaussians._semantic_confidence = gaussians._semantic_confidence[keep_mask]

    return sparse_gaussians


if __name__ == "__main__":
    print("Gaussian Sparsification Utility")
    print("Usage: Import and call sparsify_gaussians()")
