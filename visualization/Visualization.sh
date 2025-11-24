#!/bin/bash

# Robot Pose Visualization Script
# This script runs all robot visualization scripts and saves results

set -e  # Exit on error

# ===== Configuration =====
DINOV3_MODEL_TYPE="${1:-siglip2_only}" # Default to dino_conv_only if no argument is provided
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

# ===== Fr5 Visualization =====
print_header "1/5: Visualizing Fr5 Robot"
OUTPUT_FILE="$OUTPUT_DIR/fr5_visualization_${TIMESTAMP}.png"
print_info "Running visualize_fr5.py..."

if python visualization/visualize_fr5.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Fr5 visualization saved to: $OUTPUT_FILE"
else
    print_error "Fr5 visualization failed"
fi

echo ""

# ===== Franka Research 3 Visualization =====
print_header "2/5: Visualizing Franka Research 3 Robot"
OUTPUT_FILE="$OUTPUT_DIR/franka_research3_visualization_${TIMESTAMP}.png"
print_info "Running visualize_franka_research3.py..."

if python visualization/visualize_franka_research3.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Franka Research 3 visualization saved to: $OUTPUT_FILE"
else
    print_error "Franka Research 3 visualization failed"
fi

echo ""

# ===== Meca500 Visualization =====
print_header "3/5: Visualizing Meca500 Robot"
OUTPUT_FILE="$OUTPUT_DIR/meca500_visualization_${TIMESTAMP}.png"
print_info "Running visualize_meca500.py..."

if python visualization/visualize_meca500.py --checkpoint "$CHECKPOINT_PATH" --num_samples 6 --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Meca500 visualization saved to: $OUTPUT_FILE"
else
    print_error "Meca500 visualization failed"
fi

echo ""

# ===== Meca Insertion Visualization =====
print_header "4/5: Visualizing Meca Insertion Robot"
OUTPUT_FILE="$OUTPUT_DIR/meca_insertion_visualization_${TIMESTAMP}.png"
print_info "Running visualize_meca_insertion.py..."

if python visualization/visualize_meca_insertion.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Meca Insertion visualization saved to: $OUTPUT_FILE"
else
    print_error "Meca Insertion visualization failed"
fi

echo ""

# ===== Dream Visualization =====
print_header "5/5: Visualizing Dream Robot"
OUTPUT_FILE="$OUTPUT_DIR/dream_visualization_${TIMESTAMP}.png"
print_info "Running visualize_dream.py..."

if python visualization/visualize_dream.py --checkpoint "$CHECKPOINT_PATH" --output "$OUTPUT_FILE" --model_type "$DINOV3_MODEL_TYPE"; then
    print_success "Dream visualization saved to: $OUTPUT_FILE"
else
    print_error "Dream visualization failed"
fi

echo ""

# ===== Summary =====
print_header "Visualization Complete"
print_success "All visualizations completed!"
print_info "Results saved in: $OUTPUT_DIR/"
echo ""
ls -lh "$OUTPUT_DIR/"*${TIMESTAMP}.png 2>/dev/null || echo "No files generated"

echo ""
print_info "To view results, open the PNG files in: $OUTPUT_DIR/"


# ===== Summary =====
print_header "Visualization Complete"
print_success "All visualizations completed!"
print_info "Results saved in: $OUTPUT_DIR/"
echo ""
ls -lh "$OUTPUT_DIR/"*${TIMESTAMP}.png 2>/dev/null || echo "No files generated"

echo ""
print_info "To view results, open the PNG files in: $OUTPUT_DIR/"