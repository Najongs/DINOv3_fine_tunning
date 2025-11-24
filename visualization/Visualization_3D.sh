#!/bin/bash

# Robot Pose 3D Visualization Script
# This script runs all robot 3D visualization scripts and saves results

set -e  # Exit on error

# ===== Configuration =====
DINOV3_MODEL_TYPE="${1:-dino_conv_only}" # Default to dino_conv_only if no argument is provided
CHECKPOINT_PATH="./checkpoints_total_${DINOV3_MODEL_TYPE}/best_model.pth"
OUTPUT_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ===== Functions =====
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Valid DINOv3 model types
VALID_MODEL_TYPES=("combined" "dino_only" "dino_conv_only" "combined_conv" "siglip_only" "siglip_combined" "siglip2_only" "siglip2_combined")

# Check if the provided model type is valid
if [[ ! " ${VALID_MODEL_TYPES[@]} " =~ " ${DINOV3_MODEL_TYPE} " ]]; then
    print_error "Invalid DINOv3 model type: ${DINOV3_MODEL_TYPE}"
    echo "Usage: $0 [DINOV3_MODEL_TYPE] [OPTIONAL: checkpoint_path_override]"
    echo "Valid DINOv3_MODEL_TYPEs: ${VALID_MODEL_TYPES[*]}"
    exit 1
fi

# Override checkpoint path if a second argument is provided
if [ -n "$2" ]; then
    CHECKPOINT_PATH="$2"
    print_info "Overriding checkpoint path to: $CHECKPOINT_PATH"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    print_error "Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please ensure the model type is correct and training has completed, or provide a valid custom path."
    echo "Example: $0 dino_conv_only"
    echo "Example with custom path: $0 dino_conv_only ../my_custom_checkpoint/model.pth"
    exit 1
fi

print_success "Using DINOv3 Model Type: ${DINOV3_MODEL_TYPE}"
print_success "Using checkpoint: $CHECKPOINT_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

echo ""

# ===== Fr5 3D Visualization =====
print_header "1/5: Visualizing Fr5 Robot (3D)"
OUTPUT_FILE="$OUTPUT_DIR/fr5_3d_visualization_${TIMESTAMP}.png"
print_info "Running visualize_fr5_3d.py..."

if python visualization/visualize_fr5_3d.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Fr5 3D visualization saved to: $OUTPUT_FILE"
else
    print_error "Fr5 3D visualization failed"
fi

echo ""

# ===== Franka Research 3 3D Visualization =====
print_header "2/5: Visualizing Franka Research 3 Robot (3D)"
OUTPUT_FILE="$OUTPUT_DIR/franka_research3_3d_visualization_${TIMESTAMP}.png"
print_info "Running visualize_franka_research3_3d.py..."

if python visualization/visualize_franka_research3_3d.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Franka Research 3 3D visualization saved to: $OUTPUT_FILE"
else
    print_error "Franka Research 3 3D visualization failed"
fi

echo ""

# ===== Meca500 3D Visualization =====
print_header "3/5: Visualizing Meca500 Robot (3D)"
OUTPUT_FILE="$OUTPUT_DIR/meca500_3d_visualization_${TIMESTAMP}.png"
print_info "Running visualize_meca500_3d.py..."

if python visualization/visualize_meca500_3d.py --checkpoint "$CHECKPOINT_PATH" --num_samples 6 --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Meca500 3D visualization saved to: $OUTPUT_FILE"
else
    print_error "Meca500 3D visualization failed"
fi

echo ""

# ===== Meca Insertion 3D Visualization =====
print_header "4/5: Visualizing Meca Insertion Robot (3D)"
OUTPUT_FILE="$OUTPUT_DIR/meca_insertion_3d_visualization_${TIMESTAMP}.png"
print_info "Running visualize_meca_insertion_3d.py..."

if python visualization/visualize_meca_insertion_3d.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Meca Insertion 3D visualization saved to: $OUTPUT_FILE"
else
    print_error "Meca Insertion 3D visualization failed"
fi

echo ""

# ===== Dream 3D Visualization =====
print_header "5/5: Visualizing Dream/Panda Robot (3D)"
OUTPUT_FILE="$OUTPUT_DIR/dream_3d_visualization_${TIMESTAMP}.png"
print_info "Running visualize_dream_3d.py..."

if python visualization/visualize_dream_3d.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Dream 3D visualization saved to: $OUTPUT_FILE"
else
    print_error "Dream 3D visualization failed"
fi

echo ""

# ===== Summary =====
print_header "3D Visualization Complete"
print_success "All 3D visualizations completed!"
print_info "Results saved in: $OUTPUT_DIR/"
echo ""
ls -lh "$OUTPUT_DIR/"*3d_visualization_${TIMESTAMP}.png 2>/dev/null || echo "No files generated"

echo ""
print_info "To view results, open the PNG files in: $OUTPUT_DIR/"
