#!/bin/bash

# Integrated Multi-Model Pipeline
# Combines: Robot Pose + Depth + Human Pose

# =========================
# Configuration
# =========================

# Robot Pose Model
ROBOT_CHECKPOINT="checkpoints_simple_dino_only/latest_checkpoint.pth"
ROBOT_MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
ROBOT_HEATMAP_SIZE="512,512"
ROBOT_CLASS="research3"  # Options: research3, Fr5, MecaInsertion, Meca500, panda

# Depth Model
DEPTH_MODEL_NAME="depth-anything/DA3NESTED-GIANT-LARGE"

# YOLO-Pose Model (auto-download on first use)
YOLO_POSE_MODEL="yolo11l-pose.pt"  # Options: yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose (or yolov8*-pose)

# GPU Settings
USE_MULTI_GPU="--use_multi_gpu"  # Comment out to use sequential mode
ROBOT_GPU=0
DEPTH_GPU=1
HUMAN_GPU=2

# Input/Output
IMAGE_PATH="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/franka_research3_pose1/Panda_dataset_1th/view1/zed_41182735_left_1756275914.348.jpg"
OUTPUT_PATH="integrated_result.png"

# =========================
# Check Files
# =========================

echo "========================================="
echo "Integrated Multi-Model Pipeline"
echo "========================================="

if [ ! -f "$ROBOT_CHECKPOINT" ]; then
    echo "❌ Robot checkpoint not found: $ROBOT_CHECKPOINT"
    echo "Please update ROBOT_CHECKPOINT path"
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found: $IMAGE_PATH"
    echo "Please update IMAGE_PATH"
    exit 1
fi

echo "✓ Robot checkpoint: $ROBOT_CHECKPOINT"
echo "✓ Input image: $IMAGE_PATH"
echo ""

# =========================
# Run Pipeline
# =========================

echo "Running pipeline..."
echo ""

$HOME/.conda/envs/dinov3/bin/python3 integrated_pipeline.py \
    --robot_checkpoint "$ROBOT_CHECKPOINT" \
    --robot_model_name "$ROBOT_MODEL_NAME" \
    --robot_heatmap_size "$ROBOT_HEATMAP_SIZE" \
    --robot_class "$ROBOT_CLASS" \
    --depth_model_name "$DEPTH_MODEL_NAME" \
    --yolo_pose_model "$YOLO_POSE_MODEL" \
    $USE_MULTI_GPU \
    --robot_gpu $ROBOT_GPU \
    --depth_gpu $DEPTH_GPU \
    --human_gpu $HUMAN_GPU \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH"

echo ""
echo "========================================="
echo "✓ Pipeline completed!"
echo "Results saved to: $OUTPUT_PATH"
echo "========================================="
