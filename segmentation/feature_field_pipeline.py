from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 may be absent in minimal setups
    cv2 = None

from utils.dino_utils import DINOv2FeatureExtractor

from .edge_utils import LoGEdgeDetector
from .sam_refiner import SAMMaskGenerator
from .dino_cache import load_feature_from_cache, save_feature_to_cache


@dataclass
class FeatureFieldConfig:
    sam_checkpoint: str
    
    # --- DEFAULT ARGUMENTS FOLLOW ---
    sam_model_type: str = "vit_h"
    device: str = "cuda"
    sam_device: Optional[str] = None
    sam_fallback_device: str = "cpu"
    dino_model_name: str = "dinov2_vits14_reg_lc"
    dino_fallback_models: Optional[List[str]] = None
    min_mask_area: int = 512

    # === LoG EDGE DETECTION PARAMETERS ===
    log_sigma: float = 2.0
    laplacian_ksize: int = 3
    
    # === MERGING HYPERPARAMETERS (DEPTH-AWARE CONSERVATIVE) ===
    feature_weight: float = 1.0
    feature_sim_threshold: float = 0.75         # Higher threshold = less merging, more separation
    edge_strength_threshold: float = 0.25       # Lower = respect more edges
    edge_penalty: float = 0.8                   # Higher penalty = prevent merging across edges

    adjacency_dilation: int = 2                 # Smaller dilation = stricter adjacency
    min_contact_ratio: float = 0.01             # Higher = require more contact to merge
    max_merge_iterations: int = 500             # Fewer iterations
    adjacency_max_bbox_gap: int = 4             # Smaller gap = stricter adjacency
    feature_margin: float = 0.10                # Less tolerant = more strict feature matching

    # SAM kwargs for finer initial segmentation
    sam_generator_kwargs: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "points_per_side": 32,                  # More points = finer segmentation
        "pred_iou_thresh": 0.88,                # Higher = better quality masks
        "stability_score_thresh": 0.95,         # Higher = more stable masks
        "min_mask_region_area": 400,            # Lower = keep smaller regions
    })
    
    dino_max_long_edge: int = 1600
    dino_tile_size: int = 0
    dino_tile_stride: Optional[int] = None
    dino_cache_dir: Optional[str] = None
    dino_cache_precision: str = "float32"
    sam_max_long_edge: int = 2048
    
    # === DEPTH-AWARE MERGING PARAMETERS (STRICT) ===
    use_depth: bool = True
    depth_method: str = "midas"
    depth_cache_dir: Optional[str] = None
    depth_diff_threshold: float = 0.15          # LOW tolerance = prevent merging objects at different depths
    depth_boundary_threshold: float = 0.25      # Prevent merging across depth boundaries
    depth_weight: float = 0.5                   # HIGH depth influence = respect depth differences
    depth_boundary_weight: float = 1.0          # Strong penalty for crossing depth boundaries
    depth_gradient_sigma: float = 1.0           # Smaller sigma = sharper depth boundaries

@dataclass
class MaskRegion:
    mask: np.ndarray
    feature: torch.Tensor
    area: int
    bbox: Tuple[int, int, int, int]
    region_id: int
    source_ids: List[int] = field(default_factory=list)
    mean_depth: float = 0.0


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return x_min, y_min, x_max, y_max


def _binary_dilate_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy : dy + h, dx : dx + w]
    return result


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        result = _binary_dilate_once(result)
    return result


class MaskFeatureLifter:
    def __init__(
        self,
        extractor: DINOv2FeatureExtractor,
        max_long_edge: int,
        tile_size: int,
        tile_stride: Optional[int],
    ):
        self.extractor = extractor
        self.max_long_edge = max_long_edge
        self.tile_size = max(0, tile_size)
        self.tile_stride = tile_stride if tile_stride is not None else tile_size
        self.patch_size = getattr(extractor, "patch_size", 14)

    def compute_feature_map(self, image: np.ndarray) -> torch.Tensor:
        processed = self._maybe_downscale(image)
        padded, pad_h, pad_w = self._pad_to_patch_multiple(processed)

        with torch.no_grad():
            if self.tile_size > 0 and (
                padded.shape[0] > self.tile_size or padded.shape[1] > self.tile_size
            ):
                features = self._extract_features_tiled(padded)
            else:
                features = self.extractor.extract_features(padded)

        features = self._remove_feature_padding(features, pad_h, pad_w)
        result = features.detach().cpu()

        # Clear GPU memory
        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def build_regions(self, masks: Sequence[np.ndarray], feature_map: torch.Tensor, depth_map: Optional[np.ndarray] = None) -> List[MaskRegion]:
        if not masks:
            return []

        target_shape = masks[0].shape
        upsampled = self._upsample_feature_map(feature_map, target_shape)
        regions: List[MaskRegion] = []

        for idx, mask in enumerate(masks):
            area = int(mask.sum())
            if area == 0:
                continue
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(-1)
            pooled = (upsampled * mask_tensor).sum(dim=(0, 1)) / (area + 1e-6)

            # Compute mean depth for this region
            mean_depth = 0.0
            if depth_map is not None:
                mean_depth = float(depth_map[mask].mean())

            regions.append(
                MaskRegion(
                    mask=mask,
                    feature=pooled,
                    area=area,
                    bbox=_compute_bbox(mask),
                    region_id=idx,
                    source_ids=[idx],
                    mean_depth=mean_depth,
                )
            )
        return regions

    def _maybe_downscale(self, image: np.ndarray) -> np.ndarray:
        if self.max_long_edge <= 0:
            return image
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_long_edge:
            return image
        scale = self.max_long_edge / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = Image.fromarray(image).resize(new_size, Image.BICUBIC)
        return np.array(resized)

    def _pad_to_patch_multiple(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        pad_h = (math.ceil(height / self.patch_size) * self.patch_size) - height
        pad_w = (math.ceil(width / self.patch_size) * self.patch_size) - width
        if pad_h == 0 and pad_w == 0:
            return image, 0, 0
        padded = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )
        return padded, pad_h, pad_w

    def _remove_feature_padding(self, feature_map: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return feature_map
        pad_tokens_h = pad_h // self.patch_size
        pad_tokens_w = pad_w // self.patch_size
        h, w, _ = feature_map.shape
        end_h = h - pad_tokens_h if pad_tokens_h > 0 else h
        end_w = w - pad_tokens_w if pad_tokens_w > 0 else w
        return feature_map[:end_h, :end_w]

    def _extract_features_tiled(self, image: np.ndarray) -> torch.Tensor:
        tile_size = max(self.patch_size, (self.tile_size // self.patch_size) * self.patch_size)
        stride = self.tile_stride or tile_size
        stride = max(self.patch_size, (stride // self.patch_size) * self.patch_size)

        height, width = image.shape[:2]
        tile_size = min(tile_size, height, width)

        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        feature_dim = self.extractor.feature_dim

        feature_sum = torch.zeros(h_tokens, w_tokens, feature_dim, dtype=torch.float32)
        counts = torch.zeros(h_tokens, w_tokens, dtype=torch.float32)

        y_positions = self._generate_positions(height, tile_size, stride)
        x_positions = self._generate_positions(width, tile_size, stride)

        for y0 in y_positions:
            for x0 in x_positions:
                y1 = y0 + tile_size
                x1 = x0 + tile_size
                tile = image[y0:y1, x0:x1]
                tile_features = self.extractor.extract_features(tile).detach().cpu()
                th, tw, _ = tile_features.shape

                y_patch0 = y0 // self.patch_size
                x_patch0 = x0 // self.patch_size

                feature_sum[y_patch0 : y_patch0 + th, x_patch0 : x_patch0 + tw] += tile_features
                counts[y_patch0 : y_patch0 + th, x_patch0 : x_patch0 + tw] += 1.0

                # Clear memory after each tile
                del tile_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        counts = torch.clamp(counts, min=1.0).unsqueeze(-1)
        return feature_sum / counts

    @staticmethod
    def _generate_positions(length: int, tile: int, stride: int) -> List[int]:
        positions = list(range(0, max(1, length - tile + 1), stride))
        if positions[-1] != length - tile:
            positions.append(max(0, length - tile))
        return sorted(set(positions))

    @staticmethod
    def _upsample_feature_map(feature_map: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        tensor = feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, D, Hf, Wf)
        upsampled = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
        return upsampled.squeeze(0).permute(1, 2, 0)  # (H, W, D)


class HierarchicalMaskMerger:
    def __init__(
        self,
        feature_sim_threshold: float,
        edge_strength_threshold: float,
        edge_penalty: float,
        adjacency_dilation: int,
        min_contact_ratio: float,
        max_merge_iterations: int,
        depth_diff_threshold: Optional[float] = None,
        depth_boundary_threshold: Optional[float] = None,
        depth_weight: float = 0.0,
        max_bbox_gap: int = 4,
        feature_margin: float = 0.1,
        depth_boundary_weight: float = 0.0,
        feature_weight: float = 1.0,
    ) -> None:
        self.feature_sim_threshold = feature_sim_threshold
        self.edge_strength_threshold = edge_strength_threshold
        self.edge_penalty = edge_penalty
        self.edge_weight = edge_penalty  # Alias for clarity in scoring
        self.feature_weight = feature_weight
        self.adjacency_dilation = adjacency_dilation
        self.min_contact_ratio = min_contact_ratio
        self.max_merge_iterations = max_merge_iterations
        self.max_bbox_gap = max(0, max_bbox_gap)
        self.feature_margin = feature_margin
        # Depth-aware parameters
        self.depth_diff_threshold = depth_diff_threshold
        self.depth_boundary_threshold = depth_boundary_threshold
        self.depth_weight = depth_weight
        self.depth_boundary_weight = depth_boundary_weight
        self.use_depth = depth_diff_threshold is not None

    def merge(
        self,
        regions: List[MaskRegion],
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_gradients: Optional[np.ndarray] = None,
    ) -> List[MaskRegion]:
        merged_regions = list(regions)
        iteration = 0

        while iteration < self.max_merge_iterations:
            candidate = self._select_best_pair(merged_regions, edge_map, depth_map, depth_gradients)
            if candidate is None:
                break

            i, j = candidate
            new_region = self._merge_pair(merged_regions[i], merged_regions[j], depth_map)

            # Replace regions with merged result
            merged_regions.pop(max(i, j))
            merged_regions.pop(min(i, j))
            merged_regions.append(new_region)
            iteration += 1

        return merged_regions

    def _select_best_pair(
        self,
        regions: List[MaskRegion],
        edge_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_gradients: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        best_pair: Optional[Tuple[int, int]] = None
        best_score = float("-inf")

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if not self._is_adjacent(regions[i], regions[j]):
                    continue

                feature_sim = self._feature_similarity(regions[i], regions[j])
                if feature_sim < self.feature_sim_threshold - self.feature_margin:
                    continue

                boundary_mask = self._shared_boundary_mask(regions[i], regions[j])
                if boundary_mask is None:
                    continue

                boundary_strength = self._boundary_strength(boundary_mask, edge_map)
                if boundary_strength > self.edge_strength_threshold * (1.0 + self.feature_margin):
                    continue

                # Depth-aware merging
                depth_score = 0.0
                depth_boundary_penalty = 0.0
                if self.use_depth and depth_map is not None:
                    depth_consistent = self._check_depth_consistency(regions[i], regions[j], depth_map)
                    if not depth_consistent:
                        continue  # Skip this pair if depth is too different

                    depth_score = self._compute_depth_score(regions[i], regions[j], depth_map)
                    if depth_gradients is not None and self.depth_boundary_weight > 0.0:
                        depth_boundary_strength = self._depth_boundary_strength(boundary_mask, depth_gradients)
                        if self.depth_boundary_threshold is not None:
                            excess = max(0.0, depth_boundary_strength - self.depth_boundary_threshold)
                        else:
                            excess = depth_boundary_strength
                        depth_boundary_penalty = excess

                feature_component = feature_sim - self.feature_sim_threshold
                edge_component = self.edge_strength_threshold - boundary_strength
                score = self.edge_weight * edge_component + self.feature_weight * feature_component
                if self.use_depth and depth_map is not None:
                    score += self.depth_weight * depth_score
                    score -= self.depth_boundary_weight * depth_boundary_penalty

                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        if best_score <= 0:
            return None
        return best_pair

    def _is_adjacent(self, a: MaskRegion, b: MaskRegion) -> bool:
        if not self._bboxes_overlap(a.bbox, b.bbox):
            if self.max_bbox_gap == 0:
                return False
            gap = self._bbox_gap(a.bbox, b.bbox)
            return gap <= self.max_bbox_gap

        contact = self._contact_area(a.mask, b.mask)
        min_area = max(1, min(a.area, b.area))
        ratio = contact / float(min_area)
        if ratio >= self.min_contact_ratio:
            return True

        if self.max_bbox_gap > 0:
            gap = self._bbox_gap(a.bbox, b.bbox)
            return gap <= self.max_bbox_gap
        return False

    def _contact_area(self, mask_a: np.ndarray, mask_b: np.ndarray) -> int:
        dilated_a = binary_dilation(mask_a, iterations=self.adjacency_dilation)
        dilated_b = binary_dilation(mask_b, iterations=self.adjacency_dilation)
        return int(np.logical_and(dilated_a, dilated_b).sum())

    @staticmethod
    def _bboxes_overlap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> bool:
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    @staticmethod
    def _feature_similarity(a: MaskRegion, b: MaskRegion) -> float:
        return torch.nn.functional.cosine_similarity(
            a.feature.unsqueeze(0), b.feature.unsqueeze(0), eps=1e-6
        ).item()

    def _shared_boundary_mask(self, a: MaskRegion, b: MaskRegion) -> Optional[np.ndarray]:
        dilated_a = binary_dilation(a.mask, iterations=1)
        dilated_b = binary_dilation(b.mask, iterations=1)
        boundary_contact = np.logical_and(dilated_a, dilated_b)
        if boundary_contact.sum() == 0:
            return None
        return boundary_contact

    def _boundary_strength(self, contact_mask: np.ndarray, edge_map: np.ndarray) -> float:
        if contact_mask.sum() == 0:
            return 0.0
        return float(edge_map[contact_mask].mean())

    def _depth_boundary_strength(self, contact_mask: np.ndarray, depth_gradients: np.ndarray) -> float:
        if depth_gradients is None or contact_mask.sum() == 0:
            return 0.0
        return float(depth_gradients[contact_mask].mean())

    def _merge_pair(self, a: MaskRegion, b: MaskRegion, depth_map: Optional[np.ndarray] = None) -> MaskRegion:
        merged_mask = np.logical_or(a.mask, b.mask)
        total_area = max(1, a.area + b.area)
        merged_feature = (a.feature * a.area + b.feature * b.area) / total_area
        merged_bbox = _compute_bbox(merged_mask)
        merged_ids = a.source_ids + b.source_ids

        # Compute mean depth for merged region
        merged_depth = 0.0
        if depth_map is not None:
            merged_depth = (a.mean_depth * a.area + b.mean_depth * b.area) / total_area

        return MaskRegion(
            mask=merged_mask,
            feature=merged_feature,
            area=int(merged_mask.sum()),
            bbox=merged_bbox,
            region_id=min(a.region_id, b.region_id),
            source_ids=merged_ids,
            mean_depth=merged_depth,
        )

    def _check_depth_consistency(self, a: MaskRegion, b: MaskRegion, depth_map: np.ndarray) -> bool:
        """Check if two regions have consistent depth values"""
        depth_diff = abs(a.mean_depth - b.mean_depth)
        return depth_diff < self.depth_diff_threshold

    def _compute_depth_score(self, a: MaskRegion, b: MaskRegion, depth_map: np.ndarray) -> float:
        """
        Compute depth consistency score (higher = more similar depth).
        Returns score in [0, 1] range.
        """
        depth_diff = abs(a.mean_depth - b.mean_depth)
        # Exponential decay: score = exp(-depth_diff / sigma)
        sigma = self.depth_diff_threshold / 2.0
        score = np.exp(-depth_diff / sigma)
        return float(score)

    @staticmethod
    def _bbox_gap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> int:
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b
        x_gap = max(0, max(bx0 - ax1, ax0 - bx1))
        y_gap = max(0, max(by0 - ay1, ay0 - by1))
        return int(max(x_gap, y_gap))


class FeatureFieldPipeline:
    def __init__(
        self,
        config: FeatureFieldConfig,
        sam_generator: Optional[SAMMaskGenerator] = None,
        dino_extractor: Optional[DINOv2FeatureExtractor] = None,
    ) -> None:
        self.config = config
        sam_device = config.sam_device or config.device
        self.sam_generator = sam_generator or SAMMaskGenerator(
            checkpoint_path=config.sam_checkpoint,
            model_type=config.sam_model_type,
            device=sam_device,
            fallback_device=config.sam_fallback_device,
            generator_kwargs=config.sam_generator_kwargs,
            min_mask_area=config.min_mask_area,
        )
        self.edge_detector = LoGEdgeDetector(
            sigma=config.log_sigma,
            laplacian_ksize=config.laplacian_ksize
        )
        fallback_models = config.dino_fallback_models or ["dinov2_vits14_reg", "dinov2_vits14"]
        self.dino_extractor = dino_extractor or DINOv2FeatureExtractor(
            model_name=config.dino_model_name,
            device=config.device,
            fallback_models=fallback_models,
        )
        self.feature_lifter = MaskFeatureLifter(
            self.dino_extractor,
            max_long_edge=config.dino_max_long_edge,
            tile_size=config.dino_tile_size,
            tile_stride=config.dino_tile_stride,
        )
        self.feature_cache_dir = Path(config.dino_cache_dir).expanduser() if config.dino_cache_dir else None
        if self.feature_cache_dir:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sam_max_long_edge = config.sam_max_long_edge
        self.dino_cache_precision = config.dino_cache_precision

        # Initialize depth estimator or depth cache
        self.depth_estimator = None
        self.depth_cache_dir = None

        if config.use_depth:
            # Option 1: Use precomputed depth maps from cache
            if config.depth_cache_dir:
                self.depth_cache_dir = Path(config.depth_cache_dir).expanduser()
                if not self.depth_cache_dir.exists():
                    print(f"Warning: Depth cache directory not found: {config.depth_cache_dir}")
                    print("Falling back to on-the-fly depth estimation")
                    config.depth_cache_dir = None
                else:
                    print(f"✓ Using precomputed depth maps from: {config.depth_cache_dir}")

            # Option 2: Compute depth on-the-fly
            if not config.depth_cache_dir:
                try:
                    from utils.depth_estimator import DINOv2DepthEstimator
                    self.depth_estimator = DINOv2DepthEstimator(
                        method=config.depth_method,
                        device=config.device
                    )
                    print(f"✓ Depth estimation enabled ({config.depth_method})")
                except Exception as e:
                    print(f"Warning: Failed to initialize depth estimator: {e}")
                    print("Continuing without depth estimation")
                    config.use_depth = False

        self.merger = HierarchicalMaskMerger(
            feature_sim_threshold=config.feature_sim_threshold,
            edge_strength_threshold=config.edge_strength_threshold,
            edge_penalty=config.edge_penalty,
            adjacency_dilation=config.adjacency_dilation,
            min_contact_ratio=config.min_contact_ratio,
            max_merge_iterations=config.max_merge_iterations,
            depth_diff_threshold=config.depth_diff_threshold if config.use_depth else None,
            depth_boundary_threshold=config.depth_boundary_threshold if config.use_depth else None,
            depth_weight=config.depth_weight if config.use_depth else 0.0,
            max_bbox_gap=config.adjacency_max_bbox_gap,
            feature_margin=config.feature_margin,
            depth_boundary_weight=config.depth_boundary_weight if config.use_depth else 0.0,
            feature_weight=config.feature_weight,
        )

    def process_image(self, image: np.ndarray, image_name: Optional[str] = None) -> Dict[str, object]:
        """
        Runs SAM -> Canny edges -> feature lifting -> hierarchical merging on a single image.
        """
        # Clear CUDA cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sam_image, scale = self._prepare_sam_image(image)

        # Generate SAM masks with gradient disabled
        with torch.no_grad():
            sam_masks_small = self.sam_generator.generate(sam_image)

        # Clear cache after SAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sam_masks = self._rescale_masks(sam_masks_small, image.shape[:2], scale)
        if not sam_masks:
            return {
                "image_name": image_name,
                "regions": [],
                "edge_map": self.edge_detector.compute(image),
            }

        edge_map = self.edge_detector.compute(image)

        # Process DINO features with gradient disabled
        with torch.no_grad():
            feature_map = self._get_feature_map(image, image_name)

        # Clear cache after DINO
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract depth map if enabled
        depth_map = None
        if self.depth_cache_dir and image_name:
            # Load precomputed depth map
            depth_path = self.depth_cache_dir / f"{image_name}.npy"
            if depth_path.exists():
                depth_map = np.load(depth_path)
                print(f"  ✓ Loaded depth from cache: {depth_map.shape}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            else:
                print(f"  Warning: Depth map not found: {depth_path}")

        elif self.depth_estimator is not None:
            # Compute depth on-the-fly
            with torch.no_grad():
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                depth_map = self.depth_estimator.estimate(pil_image)
                print(f"  ✓ Depth extracted: {depth_map.shape}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")

            # Clear cache after depth estimation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        depth_gradients = None
        if depth_map is not None and self.config.use_depth:
            depth_gradients = self._compute_depth_gradients(depth_map)

        regions = self.feature_lifter.build_regions(sam_masks, feature_map, depth_map)
        merged_regions = self.merger.merge(regions, edge_map, depth_map, depth_gradients)

        return {
            "image_name": image_name,
            "regions": merged_regions,
            "edge_map": edge_map,
            "depth_map": depth_map,  # Add depth map to output
        }

    def precompute_dino_feature(self, image: np.ndarray, image_name: str) -> None:
        if not self.feature_cache_dir:
            raise ValueError("Feature cache directory is not configured.")
        if load_feature_from_cache(self.feature_cache_dir, image_name) is not None:
            return
        feature_map = self.feature_lifter.compute_feature_map(image)
        save_feature_to_cache(self.feature_cache_dir, image_name, feature_map, precision=self.dino_cache_precision)

    def _get_feature_map(self, image: np.ndarray, image_name: Optional[str]) -> torch.Tensor:
        if self.feature_cache_dir and image_name:
            cached = load_feature_from_cache(self.feature_cache_dir, image_name)
            if cached is not None:
                return cached
        feature_map = self.feature_lifter.compute_feature_map(image)
        if self.feature_cache_dir and image_name:
            save_feature_to_cache(self.feature_cache_dir, image_name, feature_map, precision=self.dino_cache_precision)
        return feature_map

    def _prepare_sam_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.sam_max_long_edge <= 0:
            return image, 1.0
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.sam_max_long_edge:
            return image, 1.0
        scale = self.sam_max_long_edge / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resized = Image.fromarray(image).resize(new_size, Image.BICUBIC)
        return np.array(resized), scale

    def _rescale_masks(
        self,
        masks: List[np.ndarray],
        target_shape: Tuple[int, int],
        scale: float,
    ) -> List[np.ndarray]:
        if not masks:
            return []
        if scale == 1.0:
            return masks
        target_h, target_w = target_shape
        resized = []
        for mask in masks:
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            upsampled = mask_img.resize((target_w, target_h), Image.NEAREST)
            resized.append(np.array(upsampled) > 0)
        return resized

    def _compute_depth_gradients(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute normalized depth gradients for boundary suppression."""
        sigma = max(0.0, self.config.depth_gradient_sigma)
        if cv2 is not None:
            depth_smooth = cv2.GaussianBlur(depth_map, (0, 0), sigma) if sigma > 0 else depth_map
            grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=3)
        else:
            depth_smooth = depth_map
            grad_y, grad_x = np.gradient(depth_smooth)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag -= grad_mag.min()
        denom = grad_mag.max()
        if denom < 1e-8:
            return np.zeros_like(grad_mag, dtype=np.float32)
        grad_norm = grad_mag / denom
        return grad_norm.astype(np.float32)
