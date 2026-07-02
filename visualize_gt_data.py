"""
Visualize Ground Truth (GT) data generation for Single_view_3D_Loss.py training.
Verifies that heatmaps, 3D points, joint angles, and occlusion augmentation are correctly created.
"""
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms

from dataset import (
    RobotPoseDataset,
    KeypointOcclusionAugmentor,
    robot_collate_fn,
    _scale_points,
    IMAGE_RESOLUTION,
    HEATMAP_SIZE
)
from kinematics import get_robot_kinematics


def visualize_heatmap_overlay(img_rgb, heatmaps, title="Heatmap Overlay"):
    """
    Overlay heatmaps on the original image.

    Args:
        img_rgb: Original image (H, W, 3)
        heatmaps: GT heatmaps (J, H, W)
        title: Plot title
    """
    # Sum all joint heatmaps
    combined_heatmap = heatmaps.sum(axis=0)
    combined_heatmap = (combined_heatmap / combined_heatmap.max() * 255).astype(np.uint8)

    # Resize heatmap to match image size
    h, w = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(combined_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    return overlay, combined_heatmap


def draw_skeleton(img, keypoints_2d, robot_class):
    """
    Draw skeleton connections on the image.

    Args:
        img: Image to draw on (H, W, 3)
        keypoints_2d: 2D keypoints (J, 2)
        robot_class: Robot class name
    """
    img_draw = img.copy()

    # Define skeleton connections (parent-child relationships)
    # For robotic arms, typically connect sequential joints
    connections = []
    for i in range(len(keypoints_2d) - 1):
        connections.append((i, i + 1))

    # Draw connections
    for idx1, idx2 in connections:
        pt1 = tuple(keypoints_2d[idx1].astype(int))
        pt2 = tuple(keypoints_2d[idx2].astype(int))
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for i, pt in enumerate(keypoints_2d):
        pt_int = tuple(pt.astype(int))
        cv2.circle(img_draw, pt_int, 5, (255, 0, 0), -1)
        cv2.putText(img_draw, str(i), pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img_draw


def visualize_3d_points(gt_3d_points, robot_class, title="3D Joint Positions"):
    """
    Visualize 3D joint positions in robot coordinate frame.

    Args:
        gt_3d_points: 3D points (J, 3) in robot frame
        robot_class: Robot class name
        title: Plot title
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(gt_3d_points[:, 0], gt_3d_points[:, 1], gt_3d_points[:, 2],
               c='red', marker='o', s=100, label='Joints')

    # Plot connections
    for i in range(len(gt_3d_points) - 1):
        pts = gt_3d_points[i:i+2]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)

    # Annotate points
    for i, pt in enumerate(gt_3d_points):
        ax.text(pt[0], pt[1], pt[2], f'J{i}', fontsize=8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        gt_3d_points[:, 0].max() - gt_3d_points[:, 0].min(),
        gt_3d_points[:, 1].max() - gt_3d_points[:, 1].min(),
        gt_3d_points[:, 2].max() - gt_3d_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (gt_3d_points[:, 0].max() + gt_3d_points[:, 0].min()) * 0.5
    mid_y = (gt_3d_points[:, 1].max() + gt_3d_points[:, 1].min()) * 0.5
    mid_z = (gt_3d_points[:, 2].max() + gt_3d_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return fig


def visualize_sample(dataset, idx, output_dir, with_occlusion=False):
    """
    Visualize a single sample's ground truth data.

    Args:
        dataset: RobotPoseDataset instance
        idx: Sample index
        output_dir: Directory to save visualizations
        with_occlusion: Whether to test with occlusion augmentation
    """
    # Get raw data
    (image_tensor, gt_heatmaps, gt_angles, gt_class, gt_3d_points,
     K, dist, orig_img_size, joint_confidences) = dataset[idx]

    # Convert tensors to numpy
    img_tensor_np = image_tensor.numpy()

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_denorm = img_tensor_np * std + mean
    img_denorm = np.clip(img_denorm, 0, 1)
    img_rgb = (img_denorm.transpose(1, 2, 0) * 255).astype(np.uint8)

    gt_heatmaps_np = gt_heatmaps.numpy()  # (J, H, W)
    gt_angles_np = gt_angles.numpy()
    gt_3d_points_np = gt_3d_points.numpy()
    K_np = K.numpy()
    joint_conf_np = joint_confidences.numpy()
    orig_w, orig_h = orig_img_size.numpy()

    # Get keypoint locations from heatmaps
    num_joints = gt_heatmaps_np.shape[0]
    keypoints_heatmap = []
    for j in range(num_joints):
        hm = gt_heatmaps_np[j]
        y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints_heatmap.append([x_max, y_max])
    keypoints_heatmap = np.array(keypoints_heatmap, dtype=np.float32)

    # Scale keypoints to original image size
    keypoints_img = _scale_points(keypoints_heatmap,
                                   from_size=(HEATMAP_SIZE[1], HEATMAP_SIZE[0]),
                                   to_size=(int(orig_w), int(orig_h)))

    # Resize img_rgb to original size for proper keypoint overlay
    img_rgb_orig = cv2.resize(img_rgb, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)

    # Create visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Original image with keypoints and skeleton
    ax1 = plt.subplot(2, 4, 1)
    img_with_skeleton = draw_skeleton(img_rgb_orig, keypoints_img, gt_class)
    ax1.imshow(img_with_skeleton)
    ax1.set_title(f'Original Image + Keypoints\nClass: {gt_class}', fontsize=10)
    ax1.axis('off')

    # 2. Heatmap overlay
    ax2 = plt.subplot(2, 4, 2)
    overlay, _ = visualize_heatmap_overlay(img_rgb_orig, gt_heatmaps_np)
    ax2.imshow(overlay)
    ax2.set_title('GT Heatmaps Overlay', fontsize=10)
    ax2.axis('off')

    # 3. Individual heatmaps (first 4 joints)
    for j in range(min(4, num_joints)):
        ax = plt.subplot(2, 4, 5 + j)
        ax.imshow(gt_heatmaps_np[j], cmap='hot')
        ax.set_title(f'Joint {j} Heatmap\nConf: {joint_conf_np[j]:.2f}', fontsize=9)
        ax.axis('off')

    # 4. Joint angles bar plot
    ax3 = plt.subplot(2, 4, 3)
    ax3.bar(range(len(gt_angles_np)), gt_angles_np, color='steelblue')
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_title('GT Joint Angles', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 5. 3D visualization
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    ax4.scatter(gt_3d_points_np[:, 0], gt_3d_points_np[:, 1], gt_3d_points_np[:, 2],
                c='red', marker='o', s=100)
    for i in range(len(gt_3d_points_np) - 1):
        pts = gt_3d_points_np[i:i+2]
        ax4.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
    ax4.set_xlabel('X (m)', fontsize=8)
    ax4.set_ylabel('Y (m)', fontsize=8)
    ax4.set_zlabel('Z (m)', fontsize=8)
    ax4.set_title('GT 3D Joint Positions', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = "_with_occlusion" if with_occlusion else ""
    save_path = output_path / f"gt_visualization_sample_{idx}{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\n{'='*80}")
    print(f"Sample {idx} Statistics:")
    print(f"{'='*80}")
    print(f"Robot class: {gt_class}")
    print(f"Image size (orig): {int(orig_w)} x {int(orig_h)}")
    print(f"Heatmap size: {HEATMAP_SIZE}")
    print(f"Number of joints: {num_joints}")
    print(f"Number of angles: {len(gt_angles_np)}")
    print(f"Joint confidences: {joint_conf_np}")
    print(f"GT angles (rad): {gt_angles_np}")
    print(f"GT 3D points shape: {gt_3d_points_np.shape}")
    print(f"GT 3D points range: X=[{gt_3d_points_np[:, 0].min():.3f}, {gt_3d_points_np[:, 0].max():.3f}], "
          f"Y=[{gt_3d_points_np[:, 1].min():.3f}, {gt_3d_points_np[:, 1].max():.3f}], "
          f"Z=[{gt_3d_points_np[:, 2].min():.3f}, {gt_3d_points_np[:, 2].max():.3f}]")
    print(f"Camera intrinsics K:\n{K_np}")
    print(f"Distortion coefficients: {dist.numpy()}")
    print(f"Saved visualization: {save_path.name}")
    print(f"{'='*80}\n")

    return save_path


def verify_forward_kinematics(gt_angles, gt_3d_points, robot_class):
    """
    Verify that GT 3D points are consistent with forward kinematics.

    Note: GT 3D points are in camera frame, while FK computes in robot base frame.
    We verify that FK produces valid 3D structure (not checking absolute values).

    Args:
        gt_angles: GT joint angles (A,)
        gt_3d_points: GT 3D points in camera frame (J, 3)
        robot_class: Robot class name

    Returns:
        bool: True if FK produces valid output
    """
    robot = get_robot_kinematics(robot_class)

    # Compute FK from GT angles (in robot base frame)
    angles_np = gt_angles.numpy()
    angles_truncated = robot._truncate_angles(angles_np)
    fk_3d_points = robot.forward_kinematics(angles_truncated)

    gt_3d_np = gt_3d_points.numpy()

    print(f"\nForward Kinematics Info:")
    print(f"  Number of joints: {len(fk_3d_points)}")
    print(f"  FK output shape: {fk_3d_points.shape}")
    print(f"  GT 3D points shape: {gt_3d_np.shape}")

    if fk_3d_points.shape != gt_3d_np.shape:
        print(f"  ⚠ Shape mismatch (this is expected for some robots due to excluded joints)")

    print(f"  FK 3D range (robot frame): X=[{fk_3d_points[:, 0].min():.3f}, {fk_3d_points[:, 0].max():.3f}], "
          f"Y=[{fk_3d_points[:, 1].min():.3f}, {fk_3d_points[:, 1].max():.3f}], "
          f"Z=[{fk_3d_points[:, 2].min():.3f}, {fk_3d_points[:, 2].max():.3f}]")
    print(f"  GT 3D range (camera frame): X=[{gt_3d_np[:, 0].min():.3f}, {gt_3d_np[:, 0].max():.3f}], "
          f"Y=[{gt_3d_np[:, 1].min():.3f}, {gt_3d_np[:, 1].max():.3f}], "
          f"Z=[{gt_3d_np[:, 2].min():.3f}, {gt_3d_np[:, 2].max():.3f}]")

    # Check that FK produces reasonable link lengths
    link_lengths = []
    for i in range(len(fk_3d_points) - 1):
        link_len = np.linalg.norm(fk_3d_points[i+1] - fk_3d_points[i])
        link_lengths.append(link_len)

    print(f"  Link lengths from FK: {[f'{l:.3f}' for l in link_lengths]}")

    # Verify link lengths are reasonable (between 1mm and 1m)
    if all(0.001 < l < 1.0 for l in link_lengths):
        print(f"  ✓ FK produces valid robot structure")
        return True
    else:
        print(f"  ⚠ Some link lengths seem unusual")
        return False


def main():
    # Configuration
    output_dir = "/home/najo/NAS/DIP/DINOv3_fine_tunning/gt_visualization_results"
    num_samples = 5
    test_with_occlusion = True

    print("="*80)
    print("Ground Truth Data Visualization")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"Test with occlusion: {test_with_occlusion}")
    print("="*80)

    # Setup dataset
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading dataset files...")
    json_files = glob.glob(
        "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/**/*.json",
        recursive=True
    )
    print(f"✓ Found {len(json_files)} JSON files")

    # Randomly sample files
    random.seed(42)
    if len(json_files) > num_samples:
        sampled_files = random.sample(json_files, num_samples)
    else:
        sampled_files = json_files[:num_samples]

    # Test WITHOUT occlusion augmentation first
    print(f"\n{'='*80}")
    print("Testing WITHOUT Occlusion Augmentation")
    print(f"{'='*80}")

    dataset_no_aug = RobotPoseDataset(sampled_files, transform, occlusion_augmentor=None)

    for i in range(len(dataset_no_aug)):
        print(f"\n[{i+1}/{len(dataset_no_aug)}] Visualizing sample {i}...")
        save_path = visualize_sample(dataset_no_aug, i, output_dir, with_occlusion=False)

        # Verify FK
        _, _, gt_angles, gt_class, gt_3d_points, _, _, _, _ = dataset_no_aug[i]
        verify_forward_kinematics(gt_angles, gt_3d_points, gt_class)

    # Test WITH occlusion augmentation
    if test_with_occlusion:
        print(f"\n{'='*80}")
        print("Testing WITH Occlusion Augmentation")
        print(f"{'='*80}")

        occlusion_augmentor = KeypointOcclusionAugmentor(
            prob=1.0,  # Always apply for testing
            min_occlusions=1,
            max_occlusions=3,
            min_patch_ratio=0.06,
            max_patch_ratio=0.2,
            occluded_confidence=0.15,
        )

        dataset_with_aug = RobotPoseDataset(sampled_files, transform, occlusion_augmentor=occlusion_augmentor)

        for i in range(len(dataset_with_aug)):
            print(f"\n[{i+1}/{len(dataset_with_aug)}] Visualizing sample {i} with occlusion...")
            save_path = visualize_sample(dataset_with_aug, i, output_dir, with_occlusion=True)

    print(f"\n{'='*80}")
    print("✓ Visualization completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
