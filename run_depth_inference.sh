#!/bin/bash

# Depth Estimation Inference Script
# Tests trained model on random samples from the dataset

# =========================
# Configuration
# =========================

# Checkpoint path (use best_model.pth for best results)
CHECKPOINT="checkpoints_depth_depth_dinov3/best_model.pth"

# Model configuration (should match training)
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
DEPTH_SIZE=280,504  # Height,Width from training

# Dataset paths
DEPTH_ROOT="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/depth_dataset"
SOURCE_ROOT="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"

# Inference settings
NUM_SAMPLES=20
OUTPUT_DIR="depth_inference_results"
DEVICE="cuda"

# =========================
# Run Inference
# =========================

echo "========================================="
echo "Depth Estimation Inference"
echo "========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Testing on ${NUM_SAMPLES} random samples"
echo "========================================="

$HOME/.conda/envs/dinov3/bin/python3 infer_depth.py \
    --checkpoint "${CHECKPOINT}" \
    --model_name "${MODEL_NAME}" \
    --depth_size ${DEPTH_SIZE} \
    --depth_root "${DEPTH_ROOT}" \
    --source_root "${SOURCE_ROOT}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"

echo ""
echo "✓ Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
