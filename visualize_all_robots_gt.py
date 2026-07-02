"""
Visualize GT data for all robot types in the dataset.
Creates separate visualizations for each robot class.
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


def visualize_heatmap_overlay(img_rgb, heatmaps):
    """Overlay heatmaps on the original image."""
    combined_heatmap = heatmaps.sum(axis=0)
    combined_heatmap = (combined_heatmap / combined_heatmap.max() * 255).astype(np.uint8)

    h, w = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(combined_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    return overlay


def draw_skeleton(img, keypoints_2d, robot_class):
    """Draw skeleton connections on the image."""
    img_draw = img.copy()

    # Connect sequential joints
    for i in range(len(keypoints_2d) - 1):
        pt1 = tuple(keypoints_2d[i].astype(int))
        pt2 = tuple(keypoints_2d[i + 1].astype(int))
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 3)

    # Draw keypoints
    for i, pt in enumerate(keypoints_2d):
        pt_int = tuple(pt.astype(int))
        cv2.circle(img_draw, pt_int, 7, (255, 0, 0), -1)
        cv2.putText(img_draw, str(i), (pt_int[0]+10, pt_int[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_draw


def visualize_sample(dataset, idx, output_dir, robot_name, sample_num):
    """Visualize a single sample's ground truth data."""
    (image_tensor, gt_heatmaps, gt_angles, gt_class, gt_3d_points,
     K, dist, orig_img_size, joint_confidences) = dataset[idx]

    # Denormalize image
    img_tensor_np = image_tensor.numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_denorm = img_tensor_np * std + mean
    img_denorm = np.clip(img_denorm, 0, 1)
    img_rgb = (img_denorm.transpose(1, 2, 0) * 255).astype(np.uint8)

    gt_heatmaps_np = gt_heatmaps.numpy()
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

    # Resize img_rgb to original size
    img_rgb_orig = cv2.resize(img_rgb, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_LINEAR)

    # Create visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Original image with skeleton
    ax1 = plt.subplot(2, 4, 1)
    img_with_skeleton = draw_skeleton(img_rgb_orig, keypoints_img, gt_class)
    ax1.imshow(img_with_skeleton)
    ax1.set_title(f'Original Image + Keypoints\nClass: {gt_class}', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # 2. Heatmap overlay
    ax2 = plt.subplot(2, 4, 2)
    overlay = visualize_heatmap_overlay(img_rgb_orig, gt_heatmaps_np)
    ax2.imshow(overlay)
    ax2.set_title('GT Heatmaps Overlay', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # 3. Individual heatmaps (first 4 joints)
    for j in range(min(4, num_joints)):
        ax = plt.subplot(2, 4, 5 + j)
        ax.imshow(gt_heatmaps_np[j], cmap='hot')
        ax.set_title(f'Joint {j} Heatmap\nConf: {joint_conf_np[j]:.2f}', fontsize=10)
        ax.axis('off')

    # 4. Joint angles bar plot
    ax3 = plt.subplot(2, 4, 3)
    colors = ['steelblue' if i < num_joints else 'lightgray' for i in range(len(gt_angles_np))]
    ax3.bar(range(len(gt_angles_np)), gt_angles_np, color=colors)
    ax3.set_xlabel('Joint Index', fontsize=10)
    ax3.set_ylabel('Angle (rad)', fontsize=10)
    ax3.set_title(f'GT Joint Angles\n({num_joints} DOF + gripper)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 5. 3D visualization
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    ax4.scatter(gt_3d_points_np[:, 0], gt_3d_points_np[:, 1], gt_3d_points_np[:, 2],
                c='red', marker='o', s=100)

    # Draw connections
    for i in range(len(gt_3d_points_np) - 1):
        pts = gt_3d_points_np[i:i+2]
        ax4.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)

    # Annotate points
    for i, pt in enumerate(gt_3d_points_np):
        ax4.text(pt[0], pt[1], pt[2], f'J{i}', fontsize=8)

    ax4.set_xlabel('X (m)', fontsize=9)
    ax4.set_ylabel('Y (m)', fontsize=9)
    ax4.set_zlabel('Z (m)', fontsize=9)
    ax4.set_title('GT 3D Joint Positions\n(Camera Frame)', fontsize=11, fontweight='bold')

    # Equal aspect ratio
    max_range = np.array([
        gt_3d_points_np[:, 0].max() - gt_3d_points_np[:, 0].min(),
        gt_3d_points_np[:, 1].max() - gt_3d_points_np[:, 1].min(),
        gt_3d_points_np[:, 2].max() - gt_3d_points_np[:, 2].min()
    ]).max() / 2.0

    mid_x = (gt_3d_points_np[:, 0].max() + gt_3d_points_np[:, 0].min()) * 0.5
    mid_y = (gt_3d_points_np[:, 1].max() + gt_3d_points_np[:, 1].min()) * 0.5
    mid_z = (gt_3d_points_np[:, 2].max() + gt_3d_points_np[:, 2].min()) * 0.5

    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.suptitle(f'{robot_name} Robot - Sample {sample_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{robot_name}_sample_{sample_num:02d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"\n{'='*80}")
    print(f"{robot_name} - Sample {sample_num} Statistics:")
    print(f"{'='*80}")
    print(f"Robot class: {gt_class}")
    print(f"Image size: {int(orig_w)} x {int(orig_h)}")
    print(f"Heatmap size: {HEATMAP_SIZE}")
    print(f"Number of joints: {num_joints}")
    print(f"Number of angles: {len(gt_angles_np)}")
    print(f"Joint confidences: {joint_conf_np[:num_joints]}")
    print(f"GT angles (rad): {gt_angles_np[:num_joints]}")
    print(f"GT 3D points range: X=[{gt_3d_points_np[:, 0].min():.3f}, {gt_3d_points_np[:, 0].max():.3f}], "
          f"Y=[{gt_3d_points_np[:, 1].min():.3f}, {gt_3d_points_np[:, 1].max():.3f}], "
          f"Z=[{gt_3d_points_np[:, 2].min():.3f}, {gt_3d_points_np[:, 2].max():.3f}]")

    # Verify FK
    verify_forward_kinematics(gt_angles, gt_3d_points, gt_class)

    print(f"Saved: {save_path.name}")
    print(f"{'='*80}\n")

    return save_path


def verify_forward_kinematics(gt_angles, gt_3d_points, robot_class):
    """Verify FK produces valid robot structure."""
    robot = get_robot_kinematics(robot_class)
    angles_np = gt_angles.numpy()
    angles_truncated = robot._truncate_angles(angles_np)
    fk_3d_points = robot.forward_kinematics(angles_truncated)
    gt_3d_np = gt_3d_points.numpy()

    print(f"FK Info:")
    print(f"  FK joints: {len(fk_3d_points)}, GT joints: {len(gt_3d_np)}")

    # Check link lengths
    link_lengths = []
    for i in range(len(fk_3d_points) - 1):
        link_len = np.linalg.norm(fk_3d_points[i+1] - fk_3d_points[i])
        link_lengths.append(link_len)

    print(f"  Link lengths (FK): {[f'{l:.3f}m' for l in link_lengths]}")

    # Check GT link lengths too
    gt_link_lengths = []
    for i in range(len(gt_3d_np) - 1):
        link_len = np.linalg.norm(gt_3d_np[i+1] - gt_3d_np[i])
        gt_link_lengths.append(link_len)

    print(f"  Link lengths (GT): {[f'{l:.3f}m' for l in gt_link_lengths]}")

    if all(0.001 < l < 1.5 for l in link_lengths):
        print(f"  ✓ FK produces valid structure")
    else:
        print(f"  ⚠ Some FK link lengths unusual")


def process_robot_dataset(robot_path, robot_name, output_dir, num_samples=3):
    """Process and visualize samples from a specific robot dataset."""
    print(f"\n{'#'*80}")
    print(f"# Processing {robot_name} Dataset")
    print(f"{'#'*80}")
    print(f"Path: {robot_path}")

    # Find all JSON files
    json_files = glob.glob(os.path.join(robot_path, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files")

    if len(json_files) == 0:
        print(f"⚠ No JSON files found for {robot_name}. Skipping...")
        return 0

    # Sample files
    random.seed(42)
    if len(json_files) > num_samples:
        sampled_files = random.sample(json_files, num_samples)
    else:
        sampled_files = json_files[:num_samples]
        print(f"⚠ Only {len(sampled_files)} samples available")

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset
    dataset = RobotPoseDataset(sampled_files, transform, occlusion_augmentor=None)

    # Visualize each sample
    for i in range(len(dataset)):
        print(f"\n[{i+1}/{len(dataset)}] Visualizing {robot_name} sample {i+1}...")
        try:
            visualize_sample(dataset, i, output_dir, robot_name, i+1)
        except Exception as e:
            print(f"❌ Error visualizing sample {i}: {e}")
            import traceback
            traceback.print_exc()

    return len(dataset)


def main():
    # Configuration
    base_dataset_path = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset"
    output_base_dir = "/home/najo/NAS/DIP/DINOv3_fine_tunning/gt_visualization_all_robots"
    num_samples_per_robot = 3

    # Robot datasets to process
    robot_datasets = [
        ("Fr5_to_DREAM", "Fr5"),
        ("franka_research3_to_DREAM", "Research3"),
        ("Meca_insertion_to_DREAM", "MecaInsertion"),
        ("Meca500_to_DREAM", "Meca500"),
    ]

    print("="*80)
    print("GT Data Visualization for All Robot Types")
    print("="*80)
    print(f"Base dataset path: {base_dataset_path}")
    print(f"Output directory: {output_base_dir}")
    print(f"Samples per robot: {num_samples_per_robot}")
    print("="*80)

    total_visualized = 0

    for dataset_folder, robot_name in robot_datasets:
        robot_path = os.path.join(base_dataset_path, dataset_folder)
        output_dir = os.path.join(output_base_dir, robot_name)

        if not os.path.exists(robot_path):
            print(f"\n⚠ Path does not exist: {robot_path}")
            continue

        num_visualized = process_robot_dataset(
            robot_path,
            robot_name,
            output_dir,
            num_samples_per_robot
        )
        total_visualized += num_visualized

    print(f"\n{'='*80}")
    print(f"✓ Visualization completed!")
    print(f"Total samples visualized: {total_visualized}")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
