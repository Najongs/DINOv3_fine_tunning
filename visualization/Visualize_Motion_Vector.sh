#!/bin/bash

# ======================================================
# Shell script to run the Ground Truth Motion Vector visualization
# ======================================================

# --- Configuration ---
# Number of random samples to display
NUM_SAMPLES=3

OUTPUT_DIR="results/motion_vector_viz"

# --- Script Start ---
echo "========================================"
echo "Visualizing Ground Truth Motion Vectors"
echo "========================================"
echo "→ Number of samples to show: ${NUM_SAMPLES}"
echo "→ Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Activate Conda environment if needed
# if [ -n "$CONDA_DEFAULT_ENV" ]; then
#     echo "✓ Conda environment '$CONDA_DEFAULT_ENV' is active."
# else
#     echo "⚠ Warning: No Conda environment detected. Make sure PyTorch and other dependencies are installed."
# fi

# Run the visualization script, saving outputs
python3 ./visualization/visualize_motion_vector.py --num_samples ${NUM_SAMPLES} --output "${OUTPUT_DIR}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Visualization script finished."
else
    echo ""
    echo "✗ Visualization script failed."
fi
