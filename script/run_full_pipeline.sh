#!/bin/bash

################################################################################
# Full Pipeline: Depth-Aware Feature-Field 3D Gaussian Splatting
# This script runs the complete pipeline from COLMAP to final rendering
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Required arguments
DATASET_NAME=${1:-"mipnerf360/kitchen"}
DATA_DIR=${2:-"data/${DATASET_NAME}"}
OUTPUT_BASE=${3:-"output/${DATASET_NAME}"}

# Optional configuration
GPU_ID=${GPU_ID:-1}
ITERATIONS=${ITERATIONS:-30000}
SAM_CHECKPOINT=${SAM_CHECKPOINT:-"sam_vit_h_4b8939.pth"}

# Derived paths
IMAGE_DIR="${DATA_DIR}/images"
SPARSE_DIR="${DATA_DIR}/sparse"
DEPTH_CACHE_DIR="cache/depth_maps/${DATASET_NAME}"
DINO_CACHE_DIR="cache/dino_features"
FEATURE_FIELD_DIR="${OUTPUT_BASE}/feature_field"
MODEL_DIR="${OUTPUT_BASE}/depth_aware"
CONFIG_FILE="config/gaussian_dataset/${DATASET_NAME}.json"

echo "================================================================================"
echo "Full Pipeline Execution"
echo "================================================================================"
echo "  Dataset: ${DATASET_NAME}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_BASE}"
echo "  GPU: ${GPU_ID}"
echo "  Iterations: ${ITERATIONS}"
echo "================================================================================"
echo ""

# ============================================================================
# Step 1: Verify COLMAP data
# ============================================================================

echo "================================================================================"
echo "Step 1: Verifying COLMAP data"
echo "================================================================================"

if [ ! -d "${SPARSE_DIR}" ]; then
    echo "❌ Error: COLMAP sparse directory not found: ${SPARSE_DIR}"
    echo "Please run COLMAP first or provide correct data directory"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "❌ Error: Images directory not found: ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(ls ${IMAGE_DIR}/*.{jpg,png,JPG,PNG} 2>/dev/null | wc -l)
echo "✓ Found ${NUM_IMAGES} images"
echo "✓ COLMAP data verified"
echo ""

# ============================================================================
# Step 2: Precompute depth maps
# ============================================================================

echo "================================================================================"
echo "Step 2: Precomputing depth maps"
echo "================================================================================"

if [ -d "${DEPTH_CACHE_DIR}" ] && [ "$(ls -A ${DEPTH_CACHE_DIR}/*.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "⚠ Depth cache already exists, skipping..."
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python script/precompute_depth_maps.py \
        --image-dir ${IMAGE_DIR} \
        --output-dir ${DEPTH_CACHE_DIR}
fi

echo "✓ Depth maps ready"
echo ""

# ============================================================================
# Step 3: Feature-field segmentation
# ============================================================================

echo "================================================================================"
echo "Step 3: Running feature-field segmentation (SAM + Hierarchical Merging)"
echo "================================================================================"

if [ -d "${FEATURE_FIELD_DIR}" ] && [ "$(ls -A ${FEATURE_FIELD_DIR}/*.npz 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "⚠ Feature-field already exists, skipping..."
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python script/run_feature_field_segmentation.py \
        --image-dir ${IMAGE_DIR} \
        --output-dir ${FEATURE_FIELD_DIR} \
        --sam-checkpoint ${SAM_CHECKPOINT} \
        --dino-cache-dir ${DINO_CACHE_DIR} \
        --depth-cache-dir ${DEPTH_CACHE_DIR}
fi

echo "✓ Feature-field segmentation complete"
echo ""

# ============================================================================
# Step 4: Render labeled masks for verification
# ============================================================================

echo "================================================================================"
echo "Step 4: Rendering labeled masks (2D verification)"
echo "================================================================================"

LABELED_MASK_DIR="${FEATURE_FIELD_DIR}/labeled_masks"

if [ -d "${LABELED_MASK_DIR}" ] && [ "$(ls -A ${LABELED_MASK_DIR}/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "⚠ Labeled masks already exist, skipping..."
else
    python render_masks_with_ids.py \
        --feature_field_dir ${FEATURE_FIELD_DIR} \
        --output_dir ${LABELED_MASK_DIR} \
        --font_size 14
fi

echo "✓ Labeled masks rendered"
echo ""

# ============================================================================
# Step 5: Create dataset config if not exists
# ============================================================================

echo "================================================================================"
echo "Step 5: Checking dataset configuration"
echo "================================================================================"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Creating default config file: ${CONFIG_FILE}"
    mkdir -p "$(dirname "${CONFIG_FILE}")"
    cat > ${CONFIG_FILE} <<EOF
{
    "_comment": "Config for ${DATASET_NAME} dataset",

    "densify_until_iter": 15000,
    "densify_grad_threshold": 0.0002,
    "num_classes": 256,

    "reg3d_interval": 5,
    "reg3d_k": 5,
    "reg3d_lambda_val": 2.0,
    "reg3d_max_points": 100000,
    "reg3d_sample_size": 1000,

    "opacity_reset_interval": 3000,
    "percent_dense": 0.01,
    "densification_interval": 100
}
EOF
    echo "✓ Created config: ${CONFIG_FILE}"
else
    echo "✓ Using existing config: ${CONFIG_FILE}"
fi
echo ""

# ============================================================================
# Step 6: Train 3D Gaussian Splatting with feature-field
# ============================================================================

echo "================================================================================"
echo "Step 6: Training 3D Gaussian Splatting (with Cross-view Matching)"
echo "================================================================================"

CHECKPOINT_FILE="${MODEL_DIR}/chkpnt${ITERATIONS}.pth"

if [ -f "${CHECKPOINT_FILE}" ]; then
    echo "⚠ Checkpoint already exists: ${CHECKPOINT_FILE}"
    read -p "Continue training? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training..."
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
            -s ${DATA_DIR} \
            -m ${MODEL_DIR} \
            --iterations ${ITERATIONS} \
            --feature_field_dir ${FEATURE_FIELD_DIR} \
            --config_file ${CONFIG_FILE}
    fi
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        -s ${DATA_DIR} \
        -m ${MODEL_DIR} \
        --iterations ${ITERATIONS} \
        --feature_field_dir ${FEATURE_FIELD_DIR} \
        --config_file ${CONFIG_FILE}
fi

echo "✓ Training complete"
echo ""

# ============================================================================
# Step 7: Render with feature-field visualizations
# ============================================================================

echo "================================================================================"
echo "Step 7: Rendering RGB + Feature Visualizations"
echo "================================================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python render_v2.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITERATIONS} \
    --skip_test

echo "✓ Rendering complete"
echo ""

# ============================================================================
# Step 8: Visualize 3D mask IDs
# ============================================================================

echo "================================================================================"
echo "Step 8: Rendering 3D Mask ID Visualization"
echo "================================================================================"

CUDA_VISIBLE_DEVICES=${GPU_ID} python render_object_editing.py \
    -s ${DATA_DIR} \
    -m ${MODEL_DIR} \
    --iteration ${ITERATIONS} \
    --mode visualize

echo "✓ 3D mask visualization complete"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "================================================================================"
echo "✓ Full Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Output structure:"
echo "  ${OUTPUT_BASE}/"
echo "    ├── feature_field/           # 2D refined masks + features"
echo "    │   └── labeled_masks/       # Masks with ID labels"
echo "    └── depth_aware/"
echo "        ├── point_cloud/         # Checkpoints at iterations"
echo "        ├── train/ours_${ITERATIONS}/ # Final renderings"
echo "        │   ├── renders/         # RGB images"
echo "        │   ├── gt/              # Ground truth"
echo "        │   ├── depth_maps/      # Depth visualizations"
echo "        │   ├── edge_maps/       # Edge visualizations"
echo "        │   ├── dino_features/   # DINO PCA visualizations"
echo "        │   ├── refined_masks/   # Refined SAM masks"
echo "        │   └── concat/          # Concatenated video"
echo "        └── mask_id_visualization/ # 3D mask IDs colored"
echo ""
echo "Next steps:"
echo "  1. Check labeled masks: ${LABELED_MASK_DIR}"
echo "  2. Check 3D visualization: ${MODEL_DIR}/mask_id_visualization"
echo "  3. Run object editing:"
echo "     python render_object_editing.py -s ${DATA_DIR} -m ${MODEL_DIR} \\"
echo "       --iteration ${ITERATIONS} --mode remove --mask_ids 0 1 2"
echo ""
echo "================================================================================"
