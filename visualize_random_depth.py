"""
Visualize random depth files from the generated depth dataset.
Loads random .npy depth files and their corresponding original images.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def find_all_depth_files(depth_root):
    """Find all .npy depth files recursively."""
    depth_path = Path(depth_root)
    npy_files = list(depth_path.rglob("*.npy"))
    return npy_files


def get_original_image_path(depth_path, depth_root, source_root):
    """
    Get the original image path corresponding to a depth file.

    Args:
        depth_path: Path to .npy depth file
        depth_root: Root of depth dataset
        source_root: Root of original dataset

    Returns:
        Path to original image or None if not found
    """
    # Get relative path from depth root
    rel_path = depth_path.relative_to(depth_root)

    # Try different image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_rel_path = rel_path.with_suffix(ext)
        img_path = Path(source_root) / img_rel_path
        if img_path.exists():
            return img_path

    return None


def visualize_depth_samples(depth_files, depth_root, source_root, output_dir, num_samples=5):
    """
    Visualize random depth samples with their original images.

    Args:
        depth_files: List of depth file paths
        depth_root: Root directory of depth dataset
        source_root: Root directory of original dataset
        output_dir: Directory to save visualizations
        num_samples: Number of random samples to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Randomly sample depth files
    if len(depth_files) < num_samples:
        print(f"⚠ Only {len(depth_files)} depth files available, using all")
        samples = depth_files
    else:
        samples = random.sample(depth_files, num_samples)

    print(f"\nSelected {len(samples)} random samples:")

    # Create individual visualizations
    valid_samples = []

    for i, depth_file in enumerate(samples):
        # Get original image path
        img_path = get_original_image_path(depth_file, depth_root, source_root)

        if img_path is None:
            print(f"  ❌ [{i+1}] No original image found for: {depth_file.name}")
            continue

        # Load depth
        depth = np.load(depth_file)

        # Load original image
        img = Image.open(img_path).convert('RGB')

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original image
        axes[0].imshow(img)
        rel_img_path = img_path.relative_to(source_root)
        axes[0].set_title(f'Original Image\n{rel_img_path}', fontsize=9)
        axes[0].axis('off')

        # Depth map
        im = axes[1].imshow(depth, cmap='turbo')
        axes[1].set_title(
            f'Depth Map\nmin={depth.min():.2f}, max={depth.max():.2f}, shape={depth.shape}',
            fontsize=9
        )
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save
        output_name = f'sample_{i+1}_{depth_file.stem}_vis.png'
        output_file = output_path / output_name
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        valid_samples.append((img, depth, str(rel_img_path)))
        print(f"  ✓ [{i+1}] {rel_img_path}")
        print(f"       Depth: shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")

    # Create grid visualization
    if valid_samples:
        num_valid = len(valid_samples)
        fig, axes = plt.subplots(num_valid, 2, figsize=(14, 6 * num_valid))

        if num_valid == 1:
            axes = axes.reshape(1, -1)

        for i, (img, depth, rel_path) in enumerate(valid_samples):
            # Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original: {rel_path}', fontsize=9)
            axes[i, 0].axis('off')

            # Depth map
            im = axes[i, 1].imshow(depth, cmap='turbo')
            axes[i, 1].set_title(
                f'Depth: min={depth.min():.2f}, max={depth.max():.2f}',
                fontsize=9
            )
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        grid_output = output_path / 'random_samples_grid.png'
        plt.savefig(grid_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Saved grid visualization: random_samples_grid.png")

    return len(valid_samples)


def main():
    # Configuration
    depth_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/depth_dataset"
    source_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
    output_dir = "/home/najo/NAS/DIP/DINOv3_fine_tunning/random_depth_visualization"
    num_samples = 5

    print("=" * 80)
    print("Random Depth Visualization")
    print("=" * 80)
    print(f"Depth dataset: {depth_root}")
    print(f"Source dataset: {source_root}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print("=" * 80)

    # Find all depth files
    print("\nScanning for depth files...")
    depth_files = find_all_depth_files(depth_root)
    print(f"✓ Found {len(depth_files)} depth files")

    if len(depth_files) == 0:
        print("❌ No depth files found!")
        return

    # Set random seed for reproducibility (optional)
    random.seed(42)

    # Visualize random samples
    num_visualized = visualize_depth_samples(
        depth_files,
        depth_root,
        source_root,
        output_dir,
        num_samples
    )

    print("\n" + "=" * 80)
    print(f"Visualization completed!")
    print(f"Successfully visualized: {num_visualized}/{num_samples} samples")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
