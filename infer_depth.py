"""
Inference script for trained depth estimation model.
Compares predicted depth with ground truth depth maps.
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms

from depth_model import DINOv3DepthEstimator


def find_depth_files(depth_root, num_samples=10):
    """Find random depth files from the dataset."""
    depth_path = Path(depth_root)
    npy_files = list(depth_path.rglob("*.npy"))

    if len(npy_files) == 0:
        return []

    if len(npy_files) < num_samples:
        return npy_files

    return random.sample(npy_files, num_samples)


def get_original_image_path(depth_path, depth_root, source_root):
    """Get the original image path corresponding to a depth file."""
    rel_path = depth_path.relative_to(depth_root)

    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_rel_path = rel_path.with_suffix(ext)
        img_path = Path(source_root) / img_rel_path
        if img_path.exists():
            return img_path

    return None


def compute_metrics(pred, gt, mask=None):
    """
    Compute depth estimation metrics.

    Args:
        pred: predicted depth (H, W)
        gt: ground truth depth (H, W)
        mask: optional valid mask (H, W)

    Returns:
        dict of metrics
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    else:
        pred = pred.flatten()
        gt = gt.flatten()

    # Remove invalid values
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    pred = pred[valid]
    gt = gt[valid]

    if len(pred) == 0:
        return None

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # Squared relative error
    sq_rel = np.mean(((pred - gt) ** 2) / gt)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))

    # Threshold accuracy
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25 ** 2)
    a3 = np.mean(thresh < 1.25 ** 3)

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


def visualize_comparison(img, gt_depth, pred_depth, metrics, output_path, title=""):
    """
    Visualize original image, GT depth, predicted depth, and error map.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Ground truth depth
    im1 = axes[0, 1].imshow(gt_depth, cmap='turbo')
    axes[0, 1].set_title(
        f'Ground Truth Depth\nmin={gt_depth.min():.2f}, max={gt_depth.max():.2f}',
        fontsize=11
    )
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Predicted depth
    im2 = axes[1, 0].imshow(pred_depth, cmap='turbo', vmin=gt_depth.min(), vmax=gt_depth.max())
    axes[1, 0].set_title(
        f'Predicted Depth\nmin={pred_depth.min():.2f}, max={pred_depth.max():.2f}',
        fontsize=11
    )
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Error map
    error_map = np.abs(pred_depth - gt_depth)
    im3 = axes[1, 1].imshow(error_map, cmap='hot')
    axes[1, 1].set_title(
        f'Absolute Error\nmean={error_map.mean():.3f}, max={error_map.max():.3f}',
        fontsize=11
    )
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Add metrics text
    if metrics:
        metrics_text = (
            f"Metrics:\n"
            f"AbsRel: {metrics['abs_rel']:.4f}\n"
            f"RMSE: {metrics['rmse']:.4f}\n"
            f"δ<1.25: {metrics['a1']:.3f}\n"
            f"δ<1.25²: {metrics['a2']:.3f}\n"
            f"δ<1.25³: {metrics['a3']:.3f}"
        )
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Parse depth_size
    if ',' in args.depth_size:
        depth_h, depth_w = map(int, args.depth_size.split(','))
        depth_size = (depth_h, depth_w)
    else:
        depth_size = (int(args.depth_size), int(args.depth_size))

    print("=" * 80)
    print("Depth Estimation Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Depth size: {depth_size}")
    print(f"Depth dataset: {args.depth_root}")
    print(f"Source dataset: {args.source_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device)

    # Load model
    print("\nLoading model...")
    model = DINOv3DepthEstimator(
        dino_model_name=args.model_name,
        depth_size=depth_size
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded model state dict")

    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize(depth_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find random depth files
    print(f"\nFinding {args.num_samples} random depth files...")
    depth_files = find_depth_files(args.depth_root, args.num_samples)

    if len(depth_files) == 0:
        print("❌ No depth files found!")
        return

    print(f"✓ Found {len(depth_files)} depth files")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    all_metrics = []

    for i, depth_file in enumerate(depth_files):
        print(f"\n[{i+1}/{len(depth_files)}] Processing {depth_file.name}...")

        # Get original image
        img_path = get_original_image_path(depth_file, args.depth_root, args.source_root)

        if img_path is None:
            print(f"  ❌ No original image found")
            continue

        # Load image and GT depth
        img_pil = Image.open(img_path).convert('RGB')
        gt_depth = np.load(depth_file)

        # Preprocess image
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_depth_tensor = model(img_tensor)

        # Convert to numpy
        pred_depth = pred_depth_tensor[0, 0].cpu().numpy()

        # Resize GT to match prediction if needed
        if gt_depth.shape != pred_depth.shape:
            from scipy.ndimage import zoom
            h_ratio = pred_depth.shape[0] / gt_depth.shape[0]
            w_ratio = pred_depth.shape[1] / gt_depth.shape[1]
            gt_depth_resized = zoom(gt_depth, (h_ratio, w_ratio), order=1)
        else:
            gt_depth_resized = gt_depth

        # Compute metrics
        metrics = compute_metrics(pred_depth, gt_depth_resized)

        if metrics:
            all_metrics.append(metrics)
            print(f"  Metrics: AbsRel={metrics['abs_rel']:.4f}, RMSE={metrics['rmse']:.4f}, δ<1.25={metrics['a1']:.3f}")
        else:
            print(f"  ⚠ Could not compute metrics (invalid depth values)")

        # Visualize
        rel_path = img_path.relative_to(args.source_root)
        output_path = output_dir / f"sample_{i+1:03d}_{depth_file.stem}_comparison.png"
        visualize_comparison(
            img_pil,
            gt_depth_resized,
            pred_depth,
            metrics,
            output_path,
            title=f"Sample {i+1}: {rel_path}"
        )

        print(f"  ✓ Saved visualization to {output_path.name}")

    # Print average metrics
    if all_metrics:
        print("\n" + "=" * 80)
        print("Average Metrics:")
        print("=" * 80)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

        print(f"Absolute Relative Error: {avg_metrics['abs_rel']:.4f}")
        print(f"Squared Relative Error:  {avg_metrics['sq_rel']:.4f}")
        print(f"RMSE:                    {avg_metrics['rmse']:.4f}")
        print(f"RMSE log:                {avg_metrics['rmse_log']:.4f}")
        print(f"δ < 1.25:                {avg_metrics['a1']:.3f}")
        print(f"δ < 1.25²:               {avg_metrics['a2']:.3f}")
        print(f"δ < 1.25³:               {avg_metrics['a3']:.3f}")
        print("=" * 80)

    print(f"\n✓ Inference completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Estimation Inference")

    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", help="DINOv3 model name")
    parser.add_argument("--depth_size", type=str, default="512", help="Depth map size (H,W or single value for square)")

    # Dataset paths
    parser.add_argument("--depth_root", type=str, required=True, help="Root directory of depth dataset")
    parser.add_argument("--source_root", type=str, required=True, help="Root directory of source images")

    # Inference settings
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--output_dir", type=str, default="depth_inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
