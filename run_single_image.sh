#!/bin/bash

# Single Image Test for Integrated Pipeline

# Configuration
ROBOT_CHECKPOINT="checkpoints_simple_dino_only_100e/latest_checkpoint.pth"
ROBOT_CLASS="FR5"
IMAGE_PATH="${1:-/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Fr5/Fr5_4th_250526/right/zed_34850673_right_1748249125.900.jpg}"
OUTPUT_PATH="${2:-single_image_result.png}"

# GPU settings
USE_MULTI_GPU="--use_multi_gpu"
ROBOT_GPU=0
DEPTH_GPU=1
HUMAN_GPU=2

echo "========================================"
echo "Single Image Integrated Pipeline Test"
echo "========================================"
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_PATH"
echo "========================================"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Run inference
$HOME/.conda/envs/dinov3/bin/python3 integrated_pipeline.py \
    --robot_checkpoint "$ROBOT_CHECKPOINT" \
    --robot_class "$ROBOT_CLASS" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH" \
    $USE_MULTI_GPU \
    --robot_gpu $ROBOT_GPU \
    --depth_gpu $DEPTH_GPU \
    --human_gpu $HUMAN_GPU

echo ""
echo "✓ Test complete! Result saved to: $OUTPUT_PATH"
