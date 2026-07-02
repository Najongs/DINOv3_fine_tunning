#!/bin/bash

# Human Pose Estimation Training Script
# This script trains the DINOv3-based human pose estimation model using COCO dataset

# =========================
# Configuration
# =========================

# Dataset paths
TRAIN_IMAGE_DIR="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/train2017"
TRAIN_ANNOTATION="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/annotations/person_keypoints_train2017.json"
VAL_IMAGE_DIR="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/val2017"
VAL_ANNOTATION="/home/najo/NAS/DIP/DINOv3_fine_tunning/coco_dataset/annotations/person_keypoints_val2017.json"

# Model configuration
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
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

# Set which GPUs to use (0,1,2,3,4 for all 5 GPUs, or 0,1,2,3 for 4 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Get number of available GPUs (using PyTorch to respect CUDA_VISIBLE_DEVICES)
NUM_GPUS=$($HOME/.conda/envs/dinov3/bin/python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "Found ${NUM_GPUS} GPUs available for PyTorch"

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
