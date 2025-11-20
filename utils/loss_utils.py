# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def semantic_cohesion_loss(
    gaussians_xyz,
    identity_encoding,
    dino_features,
    k=10,
    spatial_threshold=0.5,
    lambda_val=1.0,
    reliability_weights=None,
):
    
    N = gaussians_xyz.shape[0]

    max_samples = 5000  # Further reduced (Gaussians grow during training)
    if reliability_weights is not None:
        reliability_weights = reliability_weights.detach()

    if N > max_samples:
        sample_indices = torch.randperm(N, device='cuda')[:max_samples]
        sample_xyz = gaussians_xyz[sample_indices]
        sample_identity = identity_encoding[sample_indices]
        sample_dino = dino_features[sample_indices]
        sample_weights = reliability_weights[sample_indices] if reliability_weights is not None else None
    else:
        sample_xyz = gaussians_xyz
        sample_identity = identity_encoding
        sample_dino = dino_features
        sample_indices = torch.arange(N, device='cuda')
        sample_weights = reliability_weights if reliability_weights is not None else None

    N_sample = sample_xyz.shape[0]

    # 1. Find k-NN in 3D space (배치 처리로 메모리 절약)
    batch_size = 1000  # Further reduced from 2K
    topk_indices_list = []
    topk_dists_list = []

    for i in range(0, N_sample, batch_size):
        end_i = min(i + batch_size, N_sample)
        batch_xyz = sample_xyz[i:end_i]

        # Compute distances for this batch
        dists_batch = torch.cdist(batch_xyz, gaussians_xyz)  # (batch, N)

        # Get k nearest neighbors
        topk_dists_batch, topk_indices_batch = dists_batch.topk(k+1, largest=False, dim=1)
        topk_indices_list.append(topk_indices_batch[:, 1:])  # Exclude self
        topk_dists_list.append(topk_dists_batch[:, 1:])

        del dists_batch, topk_dists_batch, topk_indices_batch
        torch.cuda.empty_cache()

    topk_indices = torch.cat(topk_indices_list, dim=0)  # (N_sample, k)
    topk_dists = torch.cat(topk_dists_list, dim=0)  # (N_sample, k)

    # Filter by spatial threshold
    spatial_mask = topk_dists < spatial_threshold  # (N_sample, k)

    # 2. Compute DINO feature similarity
    # Normalize features for cosine similarity
    sample_dino_norm = F.normalize(sample_dino, p=2, dim=1)  # (N_sample, D_dino)
    all_dino_norm = F.normalize(dino_features, p=2, dim=1)  # (N, D_dino)

    # Gather neighbor DINO features
    neighbor_dino = all_dino_norm[topk_indices]  # (N_sample, k, D_dino)
    neighbor_weights = reliability_weights[topk_indices] if reliability_weights is not None else None

    # Cosine similarity with neighbors
    dino_similarity = torch.sum(
        sample_dino_norm.unsqueeze(1) * neighbor_dino,  # (N_sample, k, D_dino)
        dim=2
    )  # (N_sample, k)

    # 3. Select semantically similar neighbors
    # High DINO similarity = same object
    semantic_mask = dino_similarity > 0.7  # Threshold for semantic similarity

    # Combined mask: spatially close AND semantically similar
    valid_mask = spatial_mask & semantic_mask  # (N_sample, k)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0).cuda()

    if sample_weights is not None:
        pair_weights = sample_weights.unsqueeze(1) * neighbor_weights
    else:
        pair_weights = None

    # 4. Compute identity encoding similarity
    # Normalize identity encodings
    sample_identity_norm = F.normalize(sample_identity, p=2, dim=1)  # (N_sample, D_id)
    all_identity_norm = F.normalize(identity_encoding, p=2, dim=1)  # (N, D_id)

    # Gather neighbor identity encodings
    neighbor_identity = all_identity_norm[topk_indices]  # (N_sample, k, D_id)

    # Cosine similarity with neighbors
    identity_similarity = torch.sum(
        sample_identity_norm.unsqueeze(1) * neighbor_identity,
        dim=2
    )  # (N_sample, k)

    # 5. Loss: Encourage high identity similarity where DINO similarity is high
    # Target: identity_similarity should match dino_similarity
    # 유사한 semantic → 유사한 identity
    residual = identity_similarity[valid_mask] - dino_similarity[valid_mask]
    if pair_weights is not None:
        weight_vals = pair_weights[valid_mask]
        if weight_vals.numel() == 0 or weight_vals.sum() < 1e-8:
            return torch.tensor(0.0).cuda()
        loss = (residual ** 2 * weight_vals).sum() / (weight_vals.sum() + 1e-6)
    else:
        loss = F.mse_loss(identity_similarity[valid_mask], dino_similarity[valid_mask])

    return lambda_val * loss


def graph_connectivity_loss(gaussians_xyz, identity_encoding, dino_features,
                            predicted_ids, lambda_val=1.0):

    unique_ids = torch.unique(predicted_ids)

    total_loss = 0.0
    num_valid_clusters = 0

    for obj_id in unique_ids:
        if obj_id == 0:  # Skip background
            continue

        # Get all Gaussians with same ID
        mask = (predicted_ids == obj_id)
        cluster_size = mask.sum()

        if cluster_size < 2:  # Need at least 2 points for variance
            continue

        # Sample if cluster is too large
        if cluster_size > 1000:
            cluster_indices = torch.where(mask)[0]
            sampled_indices = cluster_indices[torch.randperm(cluster_size)[:1000]]
            mask = torch.zeros_like(mask)
            mask[sampled_indices] = True

        cluster_identity = identity_encoding[mask]
        cluster_dino = dino_features[mask]

        # 1. Identity encoding consistency within cluster
        identity_mean = cluster_identity.mean(dim=0, keepdim=True)
        loss_identity = F.mse_loss(cluster_identity, identity_mean.expand_as(cluster_identity))

        # 2. DINO feature consistency within cluster
        dino_mean = cluster_dino.mean(dim=0, keepdim=True)
        loss_dino = F.mse_loss(cluster_dino, dino_mean.expand_as(cluster_dino))

        # 3. Spatial compactness (optional, helps with connected components)
        cluster_xyz = gaussians_xyz[mask]
        xyz_center = cluster_xyz.mean(dim=0, keepdim=True)
        spatial_variance = ((cluster_xyz - xyz_center) ** 2).mean()

        # Combined loss for this cluster
        cluster_loss = loss_identity + 0.5 * loss_dino + 0.01 * spatial_variance
        total_loss += cluster_loss
        num_valid_clusters += 1

    if num_valid_clusters == 0:
        return torch.tensor(0.0).cuda()

    return lambda_val * (total_loss / num_valid_clusters)

