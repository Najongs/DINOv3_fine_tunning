#!/bin/bash

# Human Pose Estimation Inference Script
# Tests trained model on COCO dataset samples

# =========================
# Configuration
# =========================

# Checkpoint path (use best_model.pth for best results)
CHECKPOINT="checkpoints_human_pose/best_model.pth"

# Model configuration (should match training)
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_SIZE=512,512  # Height,Width from training
HEATMAP_SIZE=512,512  # Height,Width from training

# Dataset paths (COCO format)
# Update these to your actual COCO dataset paths
IMAGE_DIR="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/val2017"
ANNOTATION_FILE="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/annotations/person_keypoints_val2017.json"

# Inference settings
NUM_SAMPLES=20
OUTPUT_DIR="human_pose_inference_results"
DEVICE="cuda"

# =========================
# Run Inference
# =========================

echo "========================================="
echo "Human Pose Estimation Inference"
echo "========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Testing on ${NUM_SAMPLES} random samples"
echo "========================================="

$HOME/.conda/envs/dinov3/bin/python3 infer_human_pose.py \
    --checkpoint "${CHECKPOINT}" \
    --model_name "${MODEL_NAME}" \
    --image_size ${IMAGE_SIZE} \
    --heatmap_size ${HEATMAP_SIZE} \
    --image_dir "${IMAGE_DIR}" \
    --annotation_file "${ANNOTATION_FILE}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"

echo ""
echo "✓ Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
