#!/bin/bash

# DINOv3 Depth Estimation Training Script (Background execution)
# Runs in background and logs to file

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
NUM_GPUS=5
MASTER_PORT=29500

# Training Hyperparameters
LR=1e-4
BATCH_SIZE=16
EPOCHS=100

# Model Configuration
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
DEPTH_SIZE="280,504"
RUN_NAME="depth_dinov3"

# Log file
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/train_depth_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================"
echo "DINOv3 Depth Estimation Training (Background)"
echo "======================================================================"
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_NAME"
echo "Batch size per GPU: $BATCH_SIZE (Total: $((BATCH_SIZE * NUM_GPUS)))"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Run name: $RUN_NAME"
echo "Log file: $LOG_FILE"
echo "======================================================================"
echo ""
echo "Starting training..."

# Run training (logs will be saved to file, but output also shown on screen)
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  train_depth.py \
  --lr $LR \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --model_name $MODEL_NAME \
  --depth_size $DEPTH_SIZE \
  --run_name $RUN_NAME \
  2>&1 | tee $LOG_FILE

echo ""
echo "======================================================================"
echo "Training completed or stopped"
echo "Log saved to: $LOG_FILE"
echo "======================================================================"
