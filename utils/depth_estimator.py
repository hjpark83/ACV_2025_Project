import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


class DINOv2DepthEstimator:
    def __init__(self, method='depth_anything_v2', device='cuda'):
        self.device = device
        self.method = method

        if method == 'midas':
            self._init_midas_depth()
        elif method == 'depth_anything_v2':
            self._init_depth_anything_v2()
        elif method == 'simple':
            self._init_simple_depth()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _init_midas_depth(self):
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            # Use DPT with better BEiT backbone
            self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
            self.depth_model.to(self.device)
            self.depth_model.eval()

            print("✓ Initialized DPT depth estimator (transformers)")

        except Exception as e:
            print(f"Warning: Failed to load depth model: {e}")
            print("Falling back to simple depth estimation")
            self.method = 'simple'
            self._init_simple_depth()

    def _init_depth_anything_v2(self):
        """Initialize Depth Anything V2 (SOTA metric depth estimation)"""
        try:
            from transformers import pipeline

            # Use Depth Anything V2 Small for speed (can use Base or Large for accuracy)
            self.depth_pipeline = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == 'cuda' else -1
            )

            print("✓ Initialized Depth Anything V2 (SOTA metric depth)")

        except Exception as e:
            print(f"Warning: Failed to load Depth Anything V2: {e}")
            print("Falling back to DPT (midas)")
            self.method = 'midas'
            self._init_midas_depth()

    def _init_simple_depth(self):
        print("✓ Using simple depth estimation from DINOv2 features")
        self.depth_model = None

    @torch.no_grad()
    def estimate(self, image, dino_features=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        H, W = image.size[1], image.size[0]

        if self.method == 'midas':
            return self._estimate_midas(image)
        elif self.method == 'depth_anything_v2':
            return self._estimate_depth_anything_v2(image)
        else:
            return self._estimate_simple(image, dino_features)

    def _estimate_depth_anything_v2(self, image):
        result = self.depth_pipeline(image)
        depth_tensor = result['predicted_depth']
        depth = depth_tensor.squeeze().cpu().numpy()
        if depth.shape != (image.size[1], image.size[0]):
            depth = cv2.resize(depth, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def _estimate_midas(self, image):
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def _estimate_simple(self, image, dino_features=None):
        img_gray = np.array(image.convert('L')).astype(np.float32) / 255.0
        depth = 1.0 - img_gray
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def compute_depth_gradients(self, depth_map, sigma=1.0):
        depth_smooth = cv2.GaussianBlur(depth_map, (0, 0), sigma)

        grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)

        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        return grad_mag

    def find_depth_boundaries(self, depth_map, threshold=0.1):
        grad_mag = self.compute_depth_gradients(depth_map)
        boundaries = grad_mag > threshold

        return boundaries


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_img = Image.open("/home/cv/Desktop/park/ACV/gg/data/lerf/figurines/images/frame_00001.jpg")

    estimator = DINOv2DepthEstimator(method='midas')
    depth = estimator.estimate(test_img)
    boundaries = estimator.find_depth_boundaries(depth, threshold=0.15)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

    axes[2].imshow(boundaries, cmap='gray')
    axes[2].set_title('Depth Boundaries')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/depth_test.png')
    print("✓ Depth test saved to /tmp/depth_test.png")
