import os
import sys
import argparse
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    RobotPoseMotionVectorDataset,
    robot_motion_vector_collate_fn,
    IMAGE_RESOLUTION,
    HEATMAP_SIZE,
)
# Uncomment these when you want to add model predictions
# from model import DINOv3MotionVectorEstimator
# from utils import get_max_preds


def show_sample(sample_batch, sample_idx=0):
    """Visualizes a single sample from a batch."""
    (image_sequences, gt_heatmaps_t, gt_angles_t, gt_angle_deltas, gt_class,
     gt_3d_points_t, K, dist, joint_lengths, angle_lengths, point_lengths,
     orig_img_sizes, joint_confidences) = sample_batch

    # --- Extract data for the specific sample ---
    img_seq = image_sequences[sample_idx]
    angles_t = gt_angles_t[sample_idx].cpu().numpy()
    angle_deltas = gt_angle_deltas[sample_idx].cpu().numpy()
    num_angles = angle_lengths[sample_idx].item()

    # Trim to actual number of joints
    angles_t = angles_t[:num_angles]
    angle_deltas = angle_deltas[:num_angles]
    angles_t_plus_1 = angles_t + angle_deltas

    # --- Denormalize images for visualization ---
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_t_minus_1 = img_seq[0].cpu().permute(1, 2, 0).numpy()
    img_t_minus_1 = std * img_t_minus_1 + mean
    img_t_minus_1 = np.clip(img_t_minus_1, 0, 1)

    img_t = img_seq[1].cpu().permute(1, 2, 0).numpy()
    img_t = std * img_t + mean
    img_t = np.clip(img_t, 0, 1)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth Motion Vector Visualization (Sample #{sample_idx}, Class: {gt_class[sample_idx]})', fontsize=16)

    # Plot images
    axs[0, 0].imshow(img_t_minus_1)
    axs[0, 0].set_title('Frame t-1')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_t)
    axs[0, 1].set_title('Frame t')
    axs[0, 1].axis('off')

    # Plot angle comparisons
    joint_indices = np.arange(num_angles)
    width = 0.35

    rects1 = axs[1, 0].bar(joint_indices - width/2, angles_t, width, label='Angles (t)')
    rects2 = axs[1, 0].bar(joint_indices + width/2, angles_t_plus_1, width, label='Angles (t+1)')
    axs[1, 0].set_ylabel('Angle (rad)')
    axs[1, 0].set_title('Joint Angles: Present (t) vs Future (t+1)')
    axs[1, 0].set_xticks(joint_indices)
    axs[1, 0].set_xlabel('Joint Index')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot motion vector deltas
    axs[1, 1].bar(joint_indices, angle_deltas, color='green')
    axs[1, 1].set_ylabel('Delta (rad)')
    axs[1, 1].set_title('Motion Vector (Angle Deltas)')
    axs[1, 1].set_xticks(joint_indices)
    axs[1, 1].set_xlabel('Joint Index')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Print text summary
    print("-" * 50)
    print(f"Visualizing Sample from Class: {gt_class[sample_idx]}")
    np.set_printoptions(precision=4, suppress=True)
    print(f"GT Angles (t):       {angles_t}")
    print(f"GT Angles (t+1):     {angles_t_plus_1}")
    print(f"GT Motion (delta):   {angle_deltas}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def main(args):
    """Main function to load dataset and visualize samples."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find and sort all dataset files
    json_files = sorted(glob.glob(
        "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/**/*.json",
        recursive=True
    ))

    if not json_files:
        print("Error: No JSON files found. Check the dataset path.")
        return

    # Create the motion vector dataset
    dataset = RobotPoseMotionVectorDataset(
        json_files,
        transform,
        sequence_length=2,
        max_time_interval_s=5.0
    )

    # Use a simple dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=True,  # Shuffle to get random samples
        collate_fn=robot_motion_vector_collate_fn,
        num_workers=2
    )

    print(f"Found {len(json_files)} total files.")
    print(f"Created a dataset of size {len(dataset)}.")
    print(f"Fetching a batch of {args.num_samples} samples to visualize...")

    # Get one batch from the dataloader
    try:
        first_batch = next(iter(data_loader))
    except StopIteration:
        print("\nCould not fetch any valid samples from the dataset.")
        print("This might happen if `max_time_interval_s` is too small or there are issues with data.")
        return
        
    if first_batch is None:
        print("\nCould not fetch any valid samples from the dataset.")
        print("The collate function returned None, likely because all samples in the first batch were invalid.")
        return

    # Visualize each sample in the batch
    figures_to_show = []
    for i in range(len(first_batch[0])):
        fig = show_sample(first_batch, sample_idx=i)
        if args.output:
            output_dir = os.path.dirname(args.output)
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"motion_vector_gt_sample_{i}.png")
            fig.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_filename}")
            plt.close(fig) # Close figure to free memory
        else:
            figures_to_show.append(fig) # Store if we're showing them all at once

    if not args.output:
        plt.show() # Display all figures at once


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Motion Vectors from the dataset.")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help="Number of random samples to visualize."
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output directory to save visualization images (e.g., 'results/motion_viz'). If not specified, will display."
    )
    args = parser.parse_args()
    main(args)
