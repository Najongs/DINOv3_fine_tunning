#!/bin/bash

# Human Pose Estimation Training Script
# This script trains the DINOv3-based human pose estimation model using COCO dataset

# =========================
# Configuration
# =========================

# Dataset paths (MODIFY THESE TO YOUR ACTUAL PATHS)
TRAIN_IMAGE_DIR="/path/to/coco/images/train2017"
TRAIN_ANNOTATION="/path/to/coco/annotations/person_keypoints_train2017.json"
VAL_IMAGE_DIR="/path/to/coco/images/val2017"
VAL_ANNOTATION="/path/to/coco/annotations/person_keypoints_val2017.json"

# Model configuration
MODEL_NAME="facebook/dinov2-base"  # Options: facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
IMAGE_SIZE="512 512"  # Height Width
HEATMAP_SIZE="512 512"  # Height Width

# Training hyperparameters
LEARNING_RATE=1e-4
BATCH_SIZE=16  # Per-GPU batch size
EPOCHS=100
NUM_WORKERS=4

# Checkpointing and logging
CHECKPOINT_DIR="checkpoints_human_pose"
WANDB_PROJECT="DINOv3_HumanPose"
WANDB_RUN_NAME="human_pose_dinov2_base"

# =========================
# Multi-GPU Training
# =========================

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found ${NUM_GPUS} GPUs"

# Run distributed training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    train_human_pose.py \
    --train_image_dir "${TRAIN_IMAGE_DIR}" \
    --train_annotation_file "${TRAIN_ANNOTATION}" \
    --val_image_dir "${VAL_IMAGE_DIR}" \
    --val_annotation_file "${VAL_ANNOTATION}" \
    --model_name "${MODEL_NAME}" \
    --image_size ${IMAGE_SIZE} \
    --heatmap_size ${HEATMAP_SIZE} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}"
