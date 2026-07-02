#!/bin/bash

# Test Integrated Pipeline with Warmup and Multiple Images

# Configuration
ROBOT_CHECKPOINT="checkpoints_simple_dino_only_300e/latest_checkpoint.pth"
ROBOT_CLASS="research3"
IMAGE_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/franka_research3_pose2/Panda_dataset_22th_block/view3"
OUTPUT_DIR="integrated_test_results"
NUM_IMAGES=5  # Number of random images to test

# GPU settings (change as needed)
USE_MULTI_GPU="--use_multi_gpu"  # Comment out to disable multi-GPU
ROBOT_GPU=0
DEPTH_GPU=1
HUMAN_GPU=2

echo "========================================"
echo "Integrated Pipeline Test (Random Images)"
echo "========================================"
echo "Robot Checkpoint: $ROBOT_CHECKPOINT"
echo "Robot Class: $ROBOT_CLASS"
echo "Image Directory: $IMAGE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of Images: $NUM_IMAGES (random)"
echo "Multi-GPU: $USE_MULTI_GPU"
echo "GPUs: Robot=$ROBOT_GPU, Depth=$DEPTH_GPU, Human=$HUMAN_GPU"
echo "========================================"

# Create temporary file list with random images
TEMP_LIST=$(mktemp)
find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" \) | shuf -n $NUM_IMAGES > "$TEMP_LIST"

echo ""
echo "Selected random images:"
cat "$TEMP_LIST" | nl
echo "========================================"

# Run test
$HOME/.conda/envs/dinov3/bin/python3 test_integrated_pipeline.py \
    --robot_checkpoint "$ROBOT_CHECKPOINT" \
    --robot_class "$ROBOT_CLASS" \
    --image_list "$TEMP_LIST" \
    --output_dir "$OUTPUT_DIR" \
    $USE_MULTI_GPU \
    --robot_gpu $ROBOT_GPU \
    --depth_gpu $DEPTH_GPU \
    --human_gpu $HUMAN_GPU

# Cleanup
rm -f "$TEMP_LIST"

echo ""
echo "✓ Test complete! Results saved to: $OUTPUT_DIR"
