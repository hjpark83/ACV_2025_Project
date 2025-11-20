import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


class DINOv2FeatureExtractor:
    def __init__(self, model_name='dinov2_vits14_reg_lc', device='cuda', fallback_models=None):
        self.device = device

        self.model_name = model_name
        self.fallback_models = fallback_models or []
        loaded_model = self._load_model_with_fallback(model_name, self.fallback_models).to(device)

        if hasattr(loaded_model, 'backbone'):
            self.model = loaded_model.backbone
        elif hasattr(loaded_model, 'model'):
            self.model = loaded_model.model
        else:
            self.model = loaded_model

        self.model.eval()
        self.patch_size = 14

        self.feature_dim = 384 if 'vits' in model_name else 768

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"DINOv2 initialized: feature_dim={self.feature_dim}, patch_size={self.patch_size}")

    def _load_model_with_fallback(self, model_name, fallback_models):
        tried = []
        for candidate in [model_name] + list(fallback_models):
            try:
                print(f"Loading {candidate} from torch hub...")
                model = torch.hub.load('facebookresearch/dinov2', candidate)
                self._loaded_model_name = candidate
                return model
            except Exception as exc:
                tried.append((candidate, exc))
                print(f"⚠️  Failed to load {candidate}: {exc}")
        raise RuntimeError(f"Failed to load any DINOv2 model. Tried: {', '.join([c for c,_ in tried])}")

    @torch.no_grad()
    def extract_features(self, image, return_cls_token=False):
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            if image.max() > 1.0:
                image = image / 255.0

            image = image.to(self.device)

        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                if image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)

            if image.max() > 1.0:
                image = image / 255.0

            image = image.to(self.device)

        _, _, H, W = image.shape
        H_new = (H // self.patch_size) * self.patch_size
        W_new = (W // self.patch_size) * self.patch_size

        if H != H_new or W != W_new:
            image = F.interpolate(
                image,
                size=(H_new, W_new),
                mode='bilinear',
                align_corners=False
            )

        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        features_dict = self.model.forward_features(image)
        patch_tokens = features_dict['x_norm_patchtokens'][0]  # (N_patches, D)

        h_patch = image.shape[2] // self.patch_size
        w_patch = image.shape[3] // self.patch_size
        features = patch_tokens.reshape(h_patch, w_patch, self.feature_dim)

        if return_cls_token:
            cls_token = features_dict['x_norm_clstoken'][0]  # (D,)
            return features, cls_token

        return features

    def precompute_features_for_dataset(self, cameras, save_path=None):
        features_dict = {}

        print(f"\n{'='*60}")
        print(f"Extracting DINOv2 features for {len(cameras)} camera views")
        print(f"{'='*60}")

        for idx, camera in enumerate(cameras):
            if idx % 10 == 0:
                print(f"  [{idx:3d}/{len(cameras)}] Processing view...")

            image = camera.original_image  # torch.Tensor
            features = self.extract_features(image)  # (H_patch, W_patch, D)

            features_dict[idx] = features.cpu()

        print(f"✓ Feature extraction complete!")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Example feature shape: {features_dict[0].shape}")

        if save_path is not None:
            torch.save(features_dict, save_path)
            print(f"✓ Saved features to {save_path}")

        return features_dict


def sample_features_at_points(features_2d, points_2d, image_size):
    H_feat, W_feat, D = features_2d.shape

    features = features_2d.permute(2, 0, 1).unsqueeze(0)
    grid = points_2d.unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(
        features,
        grid,
        mode='bilinear',
        padding_mode='border',  # 이미지 경계 밖은 가장 가까운 값 사용
        align_corners=True
    )  # (1, D, N, 1)

    # 결과 reshape: (1, D, N, 1) -> (N, D)
    sampled_features = sampled.squeeze(0).squeeze(2).T

    return sampled_features


def compute_feature_variance(features_list, points_2d_list, image_sizes):
    N = points_2d_list[0].shape[0]
    D = features_list[0].shape[-1]

    all_sampled = []
    for features, points_2d, img_size in zip(features_list, points_2d_list, image_sizes):
        sampled = sample_features_at_points(features, points_2d, img_size)  # (N, D)
        all_sampled.append(sampled)

    all_sampled = torch.stack(all_sampled, dim=0)
    variance = all_sampled.var(dim=0).mean(dim=1)  # (N,)

    return variance
