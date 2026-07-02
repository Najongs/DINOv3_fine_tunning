"""
Inference script for human pose estimation model.
Tests the model on COCO dataset samples and visualizes predictions vs ground truth.
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

import torch
from torchvision import transforms

from human_pose_model import DINOv3HumanPoseEstimator
from human_pose_dataset import (
    COCOHumanPoseDataset,
    COCO_KEYPOINT_NAMES,
    NUM_KEYPOINTS
)


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12), # shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


def extract_keypoints_from_heatmaps(heatmaps, threshold=0.1):
    """
    Extract keypoint coordinates from heatmaps.

    Args:
        heatmaps: (K, H, W) heatmaps
        threshold: minimum confidence threshold

    Returns:
        keypoints: (K, 2) array of [x, y] coordinates
        confidences: (K,) array of confidence scores
    """
    K, H, W = heatmaps.shape
    keypoints = np.zeros((K, 2), dtype=np.float32)
    confidences = np.zeros(K, dtype=np.float32)

    for i in range(K):
        hm = heatmaps[i]
        max_val = hm.max()

        if max_val < threshold:
            confidences[i] = 0.0
            keypoints[i] = [0, 0]
        else:
            y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
            keypoints[i] = [x_max, y_max]
            confidences[i] = max_val

    return keypoints, confidences


def draw_skeleton(img, keypoints, confidences, skeleton_connections, color=(0, 255, 0), threshold=0.1):
    """
    Draw skeleton on image.

    Args:
        img: (H, W, 3) RGB image
        keypoints: (K, 2) keypoint coordinates
        confidences: (K,) confidence scores
        skeleton_connections: list of (idx1, idx2) pairs
        color: RGB color tuple
        threshold: minimum confidence to draw

    Returns:
        img_draw: image with skeleton drawn
    """
    img_draw = img.copy()

    # Draw connections
    for idx1, idx2 in skeleton_connections:
        if confidences[idx1] > threshold and confidences[idx2] > threshold:
            pt1 = tuple(keypoints[idx1].astype(int))
            pt2 = tuple(keypoints[idx2].astype(int))
            cv2.line(img_draw, pt1, pt2, color, 2)

    # Draw keypoints
    for i, (kpt, conf) in enumerate(zip(keypoints, confidences)):
        if conf > threshold:
            pt = tuple(kpt.astype(int))
            cv2.circle(img_draw, pt, 4, (255, 0, 0), -1)

    return img_draw


def compute_pck(pred_keypoints, gt_keypoints, gt_visibilities, threshold=0.5):
    """
    Compute Percentage of Correct Keypoints (PCK) metric.

    Args:
        pred_keypoints: (K, 2) predicted keypoint coordinates (in heatmap space)
        gt_keypoints: (K, 2) ground truth keypoint coordinates (in heatmap space)
        gt_visibilities: (K,) visibility flags (0: not visible, 1: occluded, 2: visible)
        threshold: normalized distance threshold (relative to heatmap size)

    Returns:
        pck: PCK score (percentage of correct keypoints)
        per_keypoint_correct: (K,) array of 0/1 indicating if each keypoint is correct
    """
    # Only evaluate on labeled keypoints (visibility > 0)
    labeled_mask = gt_visibilities > 0

    if not labeled_mask.any():
        return 0.0, np.zeros(len(pred_keypoints))

    # Compute Euclidean distances
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)

    # Normalize by heatmap diagonal
    heatmap_diagonal = np.sqrt(512**2 + 512**2)  # Assuming 512x512 heatmap
    normalized_distances = distances / heatmap_diagonal

    # Check which keypoints are within threshold
    correct = (normalized_distances < threshold) & labeled_mask

    # Compute PCK
    pck = correct.sum() / labeled_mask.sum()

    return pck, correct.astype(float)


def visualize_comparison(img_rgb, gt_heatmaps, pred_heatmaps, gt_keypoints, pred_keypoints,
                         gt_confidences, pred_confidences, metrics, output_path, title=""):
    """
    Visualize GT vs predicted heatmaps and keypoints.
    """
    fig = plt.figure(figsize=(18, 12))

    # 1. Original image with GT skeleton
    ax1 = plt.subplot(2, 3, 1)
    img_with_gt = draw_skeleton(img_rgb, gt_keypoints, gt_confidences, COCO_SKELETON, color=(0, 255, 0))
    ax1.imshow(img_with_gt)
    ax1.set_title('Ground Truth Skeleton', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Original image with predicted skeleton
    ax2 = plt.subplot(2, 3, 2)
    img_with_pred = draw_skeleton(img_rgb, pred_keypoints, pred_confidences, COCO_SKELETON, color=(255, 0, 0))
    ax2.imshow(img_with_pred)
    ax2.set_title('Predicted Skeleton', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. GT and predicted skeleton overlay
    ax3 = plt.subplot(2, 3, 3)
    img_overlay = img_rgb.copy()
    img_overlay = draw_skeleton(img_overlay, gt_keypoints, gt_confidences, COCO_SKELETON, color=(0, 255, 0))
    img_overlay = draw_skeleton(img_overlay, pred_keypoints, pred_confidences, COCO_SKELETON, color=(255, 0, 0))
    ax3.imshow(img_overlay)
    ax3.set_title('GT (Green) vs Predicted (Red)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. GT heatmaps (sum)
    ax4 = plt.subplot(2, 3, 4)
    gt_combined = gt_heatmaps.sum(axis=0)
    im1 = ax4.imshow(gt_combined, cmap='hot')
    ax4.set_title('GT Heatmaps (Combined)', fontsize=11)
    ax4.axis('off')
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Predicted heatmaps (sum)
    ax5 = plt.subplot(2, 3, 5)
    pred_combined = pred_heatmaps.sum(axis=0)
    im2 = ax5.imshow(pred_combined, cmap='hot')
    ax5.set_title('Predicted Heatmaps (Combined)', fontsize=11)
    ax5.axis('off')
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Per-keypoint comparison bar chart
    ax6 = plt.subplot(2, 3, 6)
    x = np.arange(NUM_KEYPOINTS)
    width = 0.35
    ax6.bar(x - width/2, gt_confidences, width, label='GT', alpha=0.7)
    ax6.bar(x + width/2, pred_confidences, width, label='Pred', alpha=0.7)
    ax6.set_xlabel('Keypoint Index', fontsize=10)
    ax6.set_ylabel('Confidence', fontsize=10)
    ax6.set_title('Per-Keypoint Confidence', fontsize=11, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(x, fontsize=7)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add metrics text
    if metrics:
        metrics_text = (
            f"PCK@0.5: {metrics['pck']:.3f}\n"
            f"Correct: {metrics['num_correct']}/{metrics['num_labeled']}"
        )
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    # Parse heatmap_size
    if ',' in args.heatmap_size:
        heatmap_h, heatmap_w = map(int, args.heatmap_size.split(','))
        heatmap_size = (heatmap_h, heatmap_w)
    else:
        heatmap_size = (int(args.heatmap_size), int(args.heatmap_size))

    # Parse image_size
    if ',' in args.image_size:
        image_h, image_w = map(int, args.image_size.split(','))
        image_size = (image_h, image_w)
    else:
        image_size = (int(args.image_size), int(args.image_size))

    print("=" * 80)
    print("Human Pose Estimation Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {image_size}")
    print(f"Heatmap size: {heatmap_size}")
    print(f"Dataset: {args.annotation_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device)

    # Load model
    print("\nLoading model...")
    model = DINOv3HumanPoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=heatmap_size,
        num_keypoints=NUM_KEYPOINTS
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

        # Filter out backbone parameters (frozen, don't need to load)
        trainable_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('backbone.model.')}

        # Load only trainable parameters with strict=False to ignore shape mismatches in frozen backbone
        missing_keys, unexpected_keys = model.load_state_dict(trainable_state_dict, strict=False)

        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

        # Only show warnings if there are missing/unexpected keys in trainable parts
        trainable_missing = [k for k in missing_keys if not k.startswith('backbone.model.')]
        trainable_unexpected = [k for k in unexpected_keys if not k.startswith('backbone.model.')]

        if trainable_missing:
            print(f"  Warning: Missing trainable keys: {trainable_missing[:5]}...")
        if trainable_unexpected:
            print(f"  Warning: Unexpected trainable keys: {trainable_unexpected[:5]}...")
    else:
        # Old format checkpoint (just state dict)
        trainable_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('backbone.model.')}
        model.load_state_dict(trainable_state_dict, strict=False)
        print("✓ Loaded model state dict")

    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = COCOHumanPoseDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        transform=transform,
        heatmap_size=heatmap_size,
        sigma=3.0,
        occlusion_augmentor=None  # No augmentation for inference
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Random sampling
    print(f"\nSampling {args.num_samples} random samples...")
    random.seed(args.seed)
    if len(dataset) < args.num_samples:
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = random.sample(range(len(dataset)), args.num_samples)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    all_pck_scores = []

    for sample_num, idx in enumerate(sample_indices):
        print(f"\n[{sample_num+1}/{len(sample_indices)}] Processing sample {idx}...")

        # Get sample
        image_tensor, gt_heatmaps, keypoint_weights = dataset[idx]

        # Get annotation info for visualization
        ann = dataset.annotations[idx]
        image_id = ann['image_id']
        img_info = dataset.images[image_id]

        # Load original image for visualization
        image_path = os.path.join(args.image_dir, img_info['file_name'])
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (heatmap_size[1], heatmap_size[0]))

        # Inference
        image_batch = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_heatmaps_tensor = model(image_batch)

        # Convert to numpy
        gt_heatmaps_np = gt_heatmaps.cpu().numpy()  # (17, H, W)
        pred_heatmaps_np = pred_heatmaps_tensor[0].cpu().numpy()  # (17, H, W)
        keypoint_weights_np = keypoint_weights.cpu().numpy()  # (17,)

        # Extract keypoints from heatmaps
        gt_keypoints, gt_confidences = extract_keypoints_from_heatmaps(gt_heatmaps_np)
        pred_keypoints, pred_confidences = extract_keypoints_from_heatmaps(pred_heatmaps_np)

        # Compute metrics
        # Convert keypoint weights to visibility (0: not visible, 2: visible)
        gt_visibilities = (keypoint_weights_np > 0).astype(int) * 2

        pck, per_keypoint_correct = compute_pck(pred_keypoints, gt_keypoints, gt_visibilities, threshold=0.5)
        num_labeled = (gt_visibilities > 0).sum()
        num_correct = per_keypoint_correct.sum()

        all_pck_scores.append(pck)

        metrics = {
            'pck': pck,
            'num_correct': int(num_correct),
            'num_labeled': int(num_labeled)
        }

        print(f"  PCK@0.5: {pck:.3f} ({num_correct}/{num_labeled} keypoints correct)")

        # Visualize
        output_path = output_dir / f"sample_{sample_num+1:03d}_idx{idx}.png"
        visualize_comparison(
            img_rgb_resized,
            gt_heatmaps_np,
            pred_heatmaps_np,
            gt_keypoints,
            pred_keypoints,
            gt_confidences,
            pred_confidences,
            metrics,
            output_path,
            title=f"Sample {sample_num+1} - Image ID: {image_id}"
        )

        print(f"  ✓ Saved visualization to {output_path.name}")

    # Print average metrics
    if all_pck_scores:
        print("\n" + "=" * 80)
        print("Average Metrics:")
        print("=" * 80)
        avg_pck = np.mean(all_pck_scores)
        print(f"Average PCK@0.5: {avg_pck:.3f}")
        print(f"Min PCK@0.5: {np.min(all_pck_scores):.3f}")
        print(f"Max PCK@0.5: {np.max(all_pck_scores):.3f}")
        print("=" * 80)

    print(f"\n✓ Inference completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Pose Estimation Inference")

    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", help="DINOv3 model name")
    parser.add_argument("--image_size", type=str, default="512", help="Image size (H,W or single value for square)")
    parser.add_argument("--heatmap_size", type=str, default="512", help="Heatmap size (H,W or single value for square)")

    # Dataset paths
    parser.add_argument("--image_dir", type=str, required=True, help="Path to COCO images directory")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO annotation JSON file")

    # Inference settings
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--output_dir", type=str, default="human_pose_inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
