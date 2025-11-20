import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


def compute_gaussian_normals(
    xyz: torch.Tensor,
    rotations: torch.Tensor,
    scalings: torch.Tensor
) -> torch.Tensor:
    
    # Convert quaternion to rotation matrix
    # q = [w, x, y, z]
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=1)
    ], dim=1)  # [N, 3, 3]

    # Smallest scale axis → normal direction
    scale_idx = scalings.argmin(dim=1)  # [N]

    # Extract corresponding rotation column
    normals = torch.zeros(len(xyz), 3, device=xyz.device)
    for i in range(len(xyz)):
        normals[i] = R[i, :, scale_idx[i]]

    # Normalize
    normals = F.normalize(normals, dim=1)

    return normals


def render_depth_map(
    camera,
    gaussians,
    bg_color: torch.Tensor = None
) -> torch.Tensor:
    
    from gaussian_renderer import render
    from arguments import PipelineParams

    if bg_color is None:
        bg_color = torch.zeros(3, device=gaussians.get_xyz.device)

    pipe = PipelineParams()

    # Render with depth
    render_pkg = render(camera, gaussians, pipe, bg_color, render_depth=True)

    if "depth" in render_pkg:
        depth_map = render_pkg["depth"]  # [1, H, W]
        return depth_map.squeeze(0)  # [H, W]
    else:
        # Fallback: compute depth from xyz
        xyz = gaussians.get_xyz
        R = camera.R if isinstance(camera.R, torch.Tensor) else torch.from_numpy(camera.R).float()
        T = camera.T if isinstance(camera.T, torch.Tensor) else torch.from_numpy(camera.T).float()
        R, T = R.to(xyz.device), T.to(xyz.device)

        # Transform to camera space
        xyz_cam = torch.matmul(xyz, R.T) + T
        depths = xyz_cam[:, 2]  # [N]

        # Project and accumulate (simplified, actual needs splatting)
        # For now, return placeholder
        H, W = camera.image_height, camera.image_width
        depth_map = torch.zeros(H, W, device=xyz.device)

        return depth_map


def render_normal_map(
    camera,
    gaussians,
    bg_color: torch.Tensor = None
) -> torch.Tensor:
    
    # Compute Gaussian normals
    xyz = gaussians.get_xyz
    rotations = gaussians.get_rotation
    scalings = gaussians.get_scaling

    normals = compute_gaussian_normals(xyz, rotations, scalings)  # [N, 3] world space

    # Transform to camera space
    R = camera.R if isinstance(camera.R, torch.Tensor) else torch.from_numpy(camera.R).float()
    R = R.to(xyz.device)

    normals_cam = torch.matmul(normals, R.T)  # [N, 3] camera space
    normals_cam = F.normalize(normals_cam, dim=1)

    # Render normal map (need custom renderer or approximate)
    # For now, create a placeholder that projects normals

    H, W = camera.image_height, camera.image_width
    normal_map = torch.zeros(3, H, W, device=xyz.device)

    # TODO: Implement actual normal splatting
    # This requires modifying the renderer to accumulate normals instead of colors

    # Placeholder: return encoded normals
    # In practice, you'd modify gaussian_renderer to support normal rendering

    return normal_map


def extract_edge_map(
    depth_map: torch.Tensor,
    threshold1: float = 50,
    threshold2: float = 150,
    use_canny: bool = True
) -> torch.Tensor:
    
    # Convert to numpy
    depth_np = depth_map.cpu().numpy()

    # Normalize depth to 0-255
    depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    if use_canny:
        # Canny edge detection
        edges = cv2.Canny(depth_uint8, threshold1, threshold2)
    else:
        # Sobel gradients
        sobelx = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)

    # Convert to tensor
    edge_map = torch.from_numpy(edges).float() / 255.0

    return edge_map.to(depth_map.device)


def extract_geometric_features(
    camera,
    gaussians,
    extract_depth: bool = True,
    extract_normal: bool = True,
    extract_edge: bool = True,
    edge_threshold1: float = 50,
    edge_threshold2: float = 150
) -> dict:
    
    features = {}

    bg_color = torch.zeros(3, device=gaussians.get_xyz.device)

    # Extract depth
    if extract_depth:
        depth_map = render_depth_map(camera, gaussians, bg_color)
        features["depth"] = depth_map
    else:
        features["depth"] = None

    # Extract normals
    if extract_normal:
        normal_map = render_normal_map(camera, gaussians, bg_color)
        features["normal"] = normal_map
    else:
        features["normal"] = None

    # Extract edges (from depth)
    if extract_edge:
        if features["depth"] is not None:
            edge_map = extract_edge_map(
                features["depth"],
                threshold1=edge_threshold1,
                threshold2=edge_threshold2
            )
            features["edge"] = edge_map
        else:
            features["edge"] = None
    else:
        features["edge"] = None

    return features


def visualize_geometric_features(
    features: dict,
    save_path: Optional[str] = None
):
    
    import matplotlib.pyplot as plt

    num_features = sum(1 for v in features.values() if v is not None)

    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))

    if num_features == 1:
        axes = [axes]

    idx = 0

    if features.get("depth") is not None:
        depth = features["depth"].cpu().numpy()
        axes[idx].imshow(depth, cmap='plasma')
        axes[idx].set_title("Depth")
        axes[idx].axis('off')
        idx += 1

    if features.get("normal") is not None:
        normal = features["normal"].cpu().numpy()
        # Convert from camera space [-1,1] to RGB [0,1]
        normal_rgb = (normal.transpose(1, 2, 0) + 1) / 2
        axes[idx].imshow(normal_rgb)
        axes[idx].set_title("Normal")
        axes[idx].axis('off')
        idx += 1

    if features.get("edge") is not None:
        edge = features["edge"].cpu().numpy()
        axes[idx].imshow(edge, cmap='gray')
        axes[idx].set_title("Edge")
        axes[idx].axis('off')
        idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("Geometric Feature Extractor")
    print("Features: Depth, Normal, Edge")
    print("Usage: Import and call extract_geometric_features()")
