#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
# Simplified training without confidence masking
# This script tests if removing complex masking improves training

ABLATION_MODE="dino_only"  # or "combined", "dino_conv_only", etc.
NUM_GPUS=5

echo "========================================="
echo "Simplified Robot Pose Training"
echo "========================================="
echo "Ablation Mode: $ABLATION_MODE"
echo "Number of GPUs: $NUM_GPUS"
echo "Changes from original:"
echo "  - No occlusion augmentation"
echo "  - No confidence masking"
echo "  - No heatmap threshold filtering"
echo "  - Simple length-based loss only"
echo "========================================="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    Single_view_3D_Loss_simple.py \
    --ablation_mode $ABLATION_MODE

echo ""
echo "========================================="
echo "✓ Training completed!"
echo "Checkpoints saved in: checkpoints_simple_$ABLATION_MODE/"
echo "========================================="
