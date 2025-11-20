from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.dino_utils import sample_features_at_points, compute_feature_variance
from segmentation.feature_field_dataset import FeatureFieldViewData


def project_points_to_camera(points_3d, camera):
    # World to camera transformation
    R = camera.R  # (3, 3)
    T = camera.T  # (3,)

    # Ensure R and T are torch tensors on GPU
    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float().cuda()
    else:
        R = R.cuda()

    if not isinstance(T, torch.Tensor):
        T = torch.from_numpy(T).float().cuda()
    else:
        T = T.cuda()

    # Transform to camera space
    # points_cam = R @ points_3d.T + T.reshape(3, 1)
    points_cam = torch.matmul(points_3d, R.T) + T  # (N, 3)

    # Extract depth (z-coordinate in camera space)
    depths = points_cam[:, 2]  # (N,)

    # Valid points are in front of the camera
    valid_mask = depths > 0.01  # Small epsilon to avoid numerical issues

    # Projection matrix
    proj_matrix = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=camera.FoVx,
        fovY=camera.FoVy
    ).cuda()  # (4, 4)

    # Homogeneous coordinates
    points_cam_h = torch.cat([points_cam, torch.ones(points_cam.shape[0], 1).cuda()], dim=1)  # (N, 4)

    # Project to clip space
    points_clip = torch.matmul(points_cam_h, proj_matrix.T)  # (N, 4)

    # Perspective divide
    points_ndc = points_clip[:, :2] / points_clip[:, 3:4]  # (N, 2)

    # NDC는 이미 [-1, 1] 범위
    points_2d = points_ndc

    return points_2d, depths, valid_mask


def compute_visibility_weights(points_3d, camera, depths, valid_mask):
    
    N = points_3d.shape[0]
    weights = torch.zeros(N).cuda()

    # 1. Visibility: valid_mask가 False이면 weight=0
    weights[~valid_mask] = 0.0

    if valid_mask.sum() == 0:
        return weights

    # 2. Depth-based weight: 가까울수록 높은 weight
    # Inverse depth with normalization
    valid_depths = depths[valid_mask]
    depth_weights = 1.0 / (valid_depths + 1e-6)
    depth_weights = depth_weights / (depth_weights.max() + 1e-6)  # Normalize to [0, 1]

    # 3. Viewing angle weight
    # Camera direction in world space
    camera_center = camera.camera_center  # (3,) world position of camera

    if not isinstance(camera_center, torch.Tensor):
        camera_center = torch.from_numpy(camera_center).float().cuda()
    else:
        camera_center = camera_center.cuda()

    view_directions = points_3d[valid_mask] - camera_center  # (N_valid, 3)
    view_directions = F.normalize(view_directions, p=2, dim=1)

    # Camera's forward direction (looking down -Z in camera space)
    R = camera.R
    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float().cuda()
    camera_forward = -R[2, :]  # (3,) third row of R matrix

    # Cosine similarity: 1 = 정면, 0 = 측면, -1 = 뒷면
    cos_angles = torch.sum(view_directions * camera_forward, dim=1)  # (N_valid,)
    angle_weights = torch.clamp(cos_angles, 0.0, 1.0)  # Ignore back-facing

    # Combine weights
    combined_weights = depth_weights * angle_weights
    weights[valid_mask] = combined_weights

    return weights


def _sample_mask_region_features(
    points_2d: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_features: torch.Tensor,
    width: int,
    height: int,
):
    if mask_features.shape[0] == 0:
        feature_dim = mask_features.shape[1] if mask_features.dim() == 2 else 0
        return (
            torch.zeros(points_2d.shape[0], feature_dim, device=points_2d.device),
            torch.zeros(points_2d.shape[0], dtype=torch.bool, device=points_2d.device),
            torch.full((points_2d.shape[0],), -1, dtype=torch.long, device=points_2d.device),
        )

    xs = ((points_2d[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5) * (width - 1)
    ys = ((points_2d[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5) * (height - 1)
    xi = torch.clamp(xs.round().long(), 0, width - 1)
    yi = torch.clamp(ys.round().long(), 0, height - 1)

    region_indices = mask_indices[yi, xi]
    has_region = region_indices >= 0
    feature_dim = mask_features.shape[1]

    sampled = torch.zeros(points_2d.shape[0], feature_dim, device=mask_features.device)
    if has_region.any():
        sampled[has_region] = mask_features[region_indices[has_region]]

    return sampled, has_region, region_indices


def lift_dino_features_to_gaussians(
    gaussians_xyz,
    cameras,
    dino_features_2d,
    variance_threshold=0.15,
    min_views=3,
    max_views=100,  
    reduce_dim=None,  
    variance_alpha=3.0,  
    log_hist_bins=10,
):
    
    N = gaussians_xyz.shape[0]
    num_views_total = len(cameras)
    feature_dim = next(iter(dino_features_2d.values())).shape[-1]

    if num_views_total > max_views:
        step = num_views_total // max_views
        sampled_view_indices = list(range(0, num_views_total, step))[:max_views]
        cameras = [cameras[i] for i in sampled_view_indices]
        dino_features_2d = {i: dino_features_2d[sampled_view_indices[i]] for i in range(len(sampled_view_indices))}
        num_views = len(cameras)
        print(f"\n⚠ Memory optimization: Using {num_views}/{num_views_total} views (every {step}th view)")
    else:
        num_views = num_views_total

    print(f"\n{'='*70}")
    print(f"Multi-view Feature Lifting (CF3-inspired)")
    print(f"{'='*70}")
    print(f"  Gaussians: {N}")
    print(f"  Views: {num_views} (total: {num_views_total})")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Variance threshold: {variance_threshold}")

    accumulated_features = torch.zeros(N, feature_dim).cuda() 
    accumulated_weights = torch.zeros(N, 1).cuda()  

    feature_sum = torch.zeros(N, feature_dim)  # E[X]
    feature_sq_sum = torch.zeros(N, feature_dim)  # E[X²]
    view_count = torch.zeros(N)  # number of views

    for view_idx, camera in enumerate(cameras):
        if view_idx % 50 == 0:
            print(f"  Processing view {view_idx}/{num_views}...")

        # Process in smaller batches to avoid OOM
        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size

        sampled_features_full = torch.zeros(N, feature_dim)
        weights_full = torch.zeros(N)  
        valid_mask_full = torch.zeros(N, dtype=torch.bool) 

        features_2d = dino_features_2d[view_idx].cuda()  # Load feature map once per view
        image_size = (camera.image_height, camera.image_width)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            # 1. Project 3D points to 2D (batch)
            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)

            # 2. Compute weights (batch)
            weights_batch = compute_visibility_weights(batch_xyz, camera, depths_batch, valid_mask_batch)

            # 3. Sample features from 2D feature map (batch)
            sampled_features_batch = sample_features_at_points(
                features_2d,
                points_2d_batch,
                image_size
            )  # (batch_size, D)

            # Store batch results in CPU tensors
            sampled_features_full[start_idx:end_idx] = sampled_features_batch.cpu()
            weights_full[start_idx:end_idx] = weights_batch.cpu()
            valid_mask_full[start_idx:end_idx] = valid_mask_batch.cpu()

            # Clean up GPU memory after each batch
            del points_2d_batch, depths_batch, valid_mask_batch, weights_batch, sampled_features_batch
            torch.cuda.empty_cache()

        valid_indices = torch.where(valid_mask_full)[0] 
        valid_features = sampled_features_full[valid_indices] 

        feature_sum[valid_indices] += valid_features
        feature_sq_sum[valid_indices] += valid_features ** 2
        view_count[valid_indices] += 1

        weighted_features = sampled_features_full * weights_full.unsqueeze(1)
        accumulated_features += weighted_features.cuda()
        accumulated_weights += weights_full.unsqueeze(1).cuda()  

        del features_2d, sampled_features_full, weights_full, valid_mask_full, weighted_features
        del valid_features, valid_indices
        torch.cuda.empty_cache()

    # 5. Normalize by total weights (already on GPU)
    gaussian_features = accumulated_features / (accumulated_weights + 1e-6)

    print(f"✓ Feature fusion complete")

    if reduce_dim is not None and reduce_dim < feature_dim:
        print(f"  Applying PCA: {feature_dim}D -> {reduce_dim}D...")

        features_cpu = gaussian_features.cpu()

        # Center the data
        mean_features = features_cpu.mean(dim=0)
        centered_features = features_cpu - mean_features

        # Compute covariance matrix (D x D)
        # Use batch processing to avoid memory issues
        cov_matrix = torch.mm(centered_features.T, centered_features) / (N - 1)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Take top reduce_dim components
        pca_components = eigenvectors[:, :reduce_dim]  # (D, reduce_dim)

        # Project features to lower dimension
        reduced_features = torch.mm(centered_features, pca_components)  # (N, reduce_dim)

        # Move back to GPU
        gaussian_features = reduced_features.cuda()
        feature_dim = reduce_dim

        # Variance explained
        variance_explained = eigenvalues[idx[:reduce_dim]].sum() / eigenvalues.sum()
        print(f"  ✓ PCA complete: {variance_explained*100:.1f}% variance retained")

        del features_cpu, mean_features, centered_features, cov_matrix
        del eigenvalues, eigenvectors, pca_components, reduced_features

    print(f"  Computing multi-view variance (incremental method)...")

    feature_mean = feature_sum / (view_count.unsqueeze(1) + 1e-6)  # (N, D)
    feature_var = (feature_sq_sum / (view_count.unsqueeze(1) + 1e-6)) - (feature_mean ** 2)  # (N, D)

    # Average variance across feature dimensions
    feature_variance = feature_var.mean(dim=1)  # (N,)
    feature_variance[view_count < min_views] = float('inf')  # Mark unreliable

    # 7. Reliable mask based on variance and visibility (on CPU)
    reliable_mask = (feature_variance < variance_threshold) & (view_count >= min_views)

    # Soft reliability weighting (higher variance -> lower weight)
    feature_variance_clamped = torch.clamp(feature_variance, min=0.0)
    reliability_weights = torch.exp(-variance_alpha * feature_variance_clamped)
    reliability_weights[view_count < min_views] = 0.0
    reliability_weights = torch.clamp(reliability_weights, 0.0, 1.0)

    feature_variance = feature_variance.cuda()
    reliability_weights = reliability_weights.cuda()
    reliable_mask = reliable_mask.cuda()

    # Statistics
    stats = {
        'total_points': N,
        'reliable_points': reliable_mask.sum().item(),
        'reliability_ratio': (reliable_mask.sum() / N).item(),
        'mean_variance': feature_variance.mean().item(),
        'mean_view_count': view_count.float().mean().item(),
        'mean_weight': reliability_weights.mean().item(),
    }

    print(f"✓ Variance filtering complete")
    print(f"  Reliable points: {stats['reliable_points']} / {stats['total_points']} "
          f"({stats['reliability_ratio']*100:.1f}%)")
    print(f"  Mean variance: {stats['mean_variance']:.4f}")
    print(f"  Mean view count: {stats['mean_view_count']:.1f}")
    print(f"  Mean reliability weight: {stats['mean_weight']:.4f}")

    variance_cpu = feature_variance.detach().cpu().numpy()
    if variance_cpu.size > 0:
        clip_max = max(variance_threshold * 3.0, variance_cpu.max() * 0.5)
        hist, bin_edges = np.histogram(np.clip(variance_cpu, 0.0, clip_max), bins=log_hist_bins)
        print("  Variance histogram (clipped):")
        for b_start, b_end, count in zip(bin_edges[:-1], bin_edges[1:], hist):
            print(f"    [{b_start:.3f}, {b_end:.3f}): {count}")
    weights_cpu = reliability_weights.cpu().detach().numpy()
    if weights_cpu.size > 0:
        sample_indices = np.linspace(0, weights_cpu.size - 1, num=min(5, weights_cpu.size), dtype=int)
        samples = [(idx, variance_cpu[idx], weights_cpu[idx]) for idx in sample_indices]
        print("  Sample (index, variance, weight):")
        for idx, var, weight in samples:
            print(f"    {idx}: var={var:.4f}, weight={weight:.4f}")

    print(f"{'='*70}\n")

    return gaussian_features, reliability_weights, stats


def lift_feature_field_masks_to_gaussians(
    gaussians_xyz,
    cameras,
    feature_field_views: Dict[str, FeatureFieldViewData],
    variance_threshold=0.12,
    min_views=2,
    variance_alpha=2.0,
    log_hist_bins=10,
):
    device = gaussians_xyz.device
    N = gaussians_xyz.shape[0]

    view_entries = []
    for idx, camera in enumerate(cameras):
        view_data = feature_field_views.get(camera.image_name)
        if view_data is None or view_data.feature_dim == 0:
            continue
        view_entries.append((idx, camera, view_data))

    if not view_entries:
        raise ValueError("No matching feature-field views were found for the provided cameras.")

    feature_dim = view_entries[0][2].feature_dim
    print(f"\n{'='*70}")
    print("Feature-field Mask Lifting")
    print(f"{'='*70}")
    print(f"  Gaussians: {N}")
    print(f"  Views with masks: {len(view_entries)} / {len(cameras)}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Variance threshold: {variance_threshold}")

    print(f"\n{'='*70}")
    print("Building cross-view mask correspondence")
    print(f"{'='*70}")

    # Collect all mask features from all views
    all_mask_descriptors = []  # List of (view_idx, local_mask_id, feature_vector)
    for view_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        tensors = view_data.to_device('cpu')
        mask_features = tensors["features"]  # (num_masks, feature_dim)
        for local_mask_id in range(mask_features.shape[0]):
            all_mask_descriptors.append({
                'view_idx': view_idx,
                'cam_idx': cam_idx,
                'local_mask_id': local_mask_id,
                'feature': mask_features[local_mask_id].cpu()
            })

    print(f"  Total masks across all views: {len(all_mask_descriptors)}")

    similarity_threshold = 0.55
    same_view_threshold = 0.65

    local_to_global_map = {}  # (view_idx, local_mask_id) -> global_mask_id
    global_mask_id = 0
    assigned = set()

    print(f"  Using ultra-aggressive thresholds:")
    print(f"    Cross-view: {similarity_threshold}")
    print(f"    Same-view: {same_view_threshold}")

    for i, desc_i in enumerate(all_mask_descriptors):
        key_i = (desc_i['view_idx'], desc_i['local_mask_id'])
        if key_i in assigned:
            continue

        # Start new cluster
        cluster_members = [key_i]
        assigned.add(key_i)

        # Find all similar masks (including from SAME view for part merging)
        feat_i = desc_i['feature']
        for j, desc_j in enumerate(all_mask_descriptors):
            if i == j:
                continue
            key_j = (desc_j['view_idx'], desc_j['local_mask_id'])
            if key_j in assigned:
                continue

            # Compute cosine similarity
            feat_j = desc_j['feature']
            similarity = torch.nn.functional.cosine_similarity(
                feat_i.unsqueeze(0), feat_j.unsqueeze(0)
            ).item()

            # For SAME view: slightly higher threshold but still aggressive
            # For DIFFERENT view: very low threshold for cross-view matching
            same_view = (desc_i['view_idx'] == desc_j['view_idx'])
            threshold = same_view_threshold if same_view else similarity_threshold

            if similarity > threshold:
                cluster_members.append(key_j)
                assigned.add(key_j)

        # Assign global ID to all members of this cluster
        for member_key in cluster_members:
            local_to_global_map[member_key] = global_mask_id

        global_mask_id += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(all_mask_descriptors)} masks, {global_mask_id} global objects found")

    print(f"  ✓ Cross-view matching complete")
    print(f"  ✓ Found {global_mask_id} globally consistent objects")
    print(f"  ✓ Matched {len(assigned)}/{len(all_mask_descriptors)} masks ({len(assigned)/len(all_mask_descriptors)*100:.1f}%)")
    print(f"{'='*70}\n")

    accumulated_features = torch.zeros(N, feature_dim, device=device)
    accumulated_weights = torch.zeros(N, 1, device=device)

    feature_sum = torch.zeros(N, feature_dim)
    feature_sq_sum = torch.zeros(N, feature_dim)
    view_count = torch.zeros(N)

    mask_id_votes = {}  # gaussian_idx -> {global_mask_id: count}

    for local_idx, (cam_idx, camera, view_data) in enumerate(view_entries):
        if local_idx % 20 == 0:
            print(f"  Processing view {local_idx}/{len(view_entries)} (camera idx {cam_idx})...")

        tensors_on_device = view_data.to_device(device)
        mask_indices = tensors_on_device["mask_indices"].long()
        mask_features = tensors_on_device["features"]

        if mask_features.shape[0] == 0:
            continue

        batch_size = 30000
        num_batches = (N + batch_size - 1) // batch_size

        sampled_features_full = torch.zeros(N, feature_dim)
        weights_full = torch.zeros(N)
        valid_mask_full = torch.zeros(N, dtype=torch.bool)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N)
            batch_xyz = gaussians_xyz[start_idx:end_idx]

            points_2d_batch, depths_batch, valid_mask_batch = project_points_to_camera(batch_xyz, camera)

            sampled_features_batch, has_region, region_ids_batch = _sample_mask_region_features(
                points_2d_batch, mask_indices, mask_features, view_data.width, view_data.height
            )
            valid_mask_batch = valid_mask_batch & has_region

            valid_batch_indices = torch.where(valid_mask_batch)[0]
            for local_idx_in_batch in valid_batch_indices:
                gaussian_idx = (start_idx + local_idx_in_batch).item()
                local_mask_id = region_ids_batch[local_idx_in_batch].item()

                # Convert local mask ID to global mask ID
                view_idx = local_idx  # This is the view index in view_entries
                key = (view_idx, local_mask_id)
                global_mask_id = local_to_global_map.get(key, -1)

                # Skip if no global mapping found
                if global_mask_id < 0:
                    continue

                if gaussian_idx not in mask_id_votes:
                    mask_id_votes[gaussian_idx] = {}
                if global_mask_id not in mask_id_votes[gaussian_idx]:
                    mask_id_votes[gaussian_idx][global_mask_id] = 0
                mask_id_votes[gaussian_idx][global_mask_id] += 1

            if valid_mask_batch.any():
                weights_batch = compute_visibility_weights(batch_xyz, camera, depths_batch, valid_mask_batch)
            else:
                weights_batch = torch.zeros_like(valid_mask_batch, dtype=torch.float32, device=device)

            sampled_features_full[start_idx:end_idx] = sampled_features_batch.detach().cpu()
            weights_full[start_idx:end_idx] = weights_batch.detach().cpu()
            valid_mask_full[start_idx:end_idx] = valid_mask_batch.detach().cpu()

            del points_2d_batch, depths_batch, valid_mask_batch, sampled_features_batch, has_region, weights_batch
            torch.cuda.empty_cache()

        valid_indices = torch.where(valid_mask_full)[0]
        if valid_indices.numel() == 0:
            continue

        valid_features = sampled_features_full[valid_indices]
        feature_sum[valid_indices] += valid_features
        feature_sq_sum[valid_indices] += valid_features**2
        view_count[valid_indices] += 1

        weighted_features = sampled_features_full * weights_full.unsqueeze(1)
        accumulated_features += weighted_features.cuda()
        accumulated_weights += weights_full.unsqueeze(1).cuda()

        del sampled_features_full, weights_full, valid_mask_full, valid_features, valid_indices, weighted_features
        torch.cuda.empty_cache()

    gaussian_features = accumulated_features / (accumulated_weights + 1e-6)

    print("✓ Feature fusion complete (mask-based)")
    print(f"  Assigning 3D mask IDs via multi-view voting...")
    gaussian_mask_ids = torch.full((N,), -1, dtype=torch.long)

    for gaussian_idx, votes in mask_id_votes.items():
        if votes:
            # Get mask ID with most votes
            best_mask_id = max(votes, key=votes.get)
            gaussian_mask_ids[gaussian_idx] = best_mask_id

    num_assigned = (gaussian_mask_ids >= 0).sum().item()
    num_unique_masks = len(torch.unique(gaussian_mask_ids[gaussian_mask_ids >= 0]))
    print(f"  ✓ Assigned 3D mask IDs: {num_assigned}/{N} Gaussians ({num_assigned/N*100:.1f}%)")
    print(f"  ✓ Unique 3D regions: {num_unique_masks}")

    print(f"  Computing multi-view variance (incremental method)...")

    feature_mean = feature_sum / (view_count.unsqueeze(1) + 1e-6)
    feature_var = (feature_sq_sum / (view_count.unsqueeze(1) + 1e-6)) - (feature_mean**2)
    feature_variance = feature_var.mean(dim=1)
    feature_variance[view_count < min_views] = float("inf")

    reliable_mask = (feature_variance < variance_threshold) & (view_count >= min_views)
    feature_variance_clamped = torch.clamp(feature_variance, min=0.0)
    reliability_weights = torch.exp(-variance_alpha * feature_variance_clamped)
    reliability_weights[view_count < min_views] = 0.0
    reliability_weights = torch.clamp(reliability_weights, 0.0, 1.0)

    feature_variance = feature_variance.cuda()
    reliability_weights = reliability_weights.cuda()
    reliable_mask = reliable_mask.cuda()

    stats = {
        "total_points": N,
        "reliable_points": reliable_mask.sum().item(),
        "reliability_ratio": (reliable_mask.sum() / N).item(),
        "mean_variance": feature_variance.mean().item(),
        "mean_view_count": view_count.float().mean().item(),
        "mean_weight": reliability_weights.mean().item(),
    }

    print(f"✓ Variance filtering complete")
    print(f"  Reliable points: {stats['reliable_points']} / {stats['total_points']} "
          f"({stats['reliability_ratio']*100:.1f}%)")
    print(f"  Mean variance: {stats['mean_variance']:.4f}")
    print(f"  Mean view count: {stats['mean_view_count']:.1f}")
    print(f"  Mean reliability weight: {stats['mean_weight']:.4f}")

    variance_cpu = feature_variance.detach().cpu().numpy()
    if variance_cpu.size > 0:
        clip_max = max(variance_threshold * 3.0, variance_cpu.max() * 0.5)
        hist, bin_edges = np.histogram(np.clip(variance_cpu, 0.0, clip_max), bins=log_hist_bins)
        print("  Variance histogram (clipped):")
        for b_start, b_end, count in zip(bin_edges[:-1], bin_edges[1:], hist):
            print(f"    [{b_start:.3f}, {b_end:.3f}): {count}")

    weights_cpu = reliability_weights.cpu().detach().numpy()
    if weights_cpu.size > 0:
        sample_indices = np.linspace(0, weights_cpu.size - 1, num=min(5, weights_cpu.size), dtype=int)
        samples = [(idx, variance_cpu[idx], weights_cpu[idx]) for idx in sample_indices]
        print("  Sample (index, variance, weight):")
        for idx, var, weight in samples:
            print(f"    {idx}: var={var:.4f}, weight={weight:.4f}")

    print(f"{'='*70}\n")
    return gaussian_features, reliability_weights, stats, gaussian_mask_ids
