import os
import json
import math
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torchvision import transforms
import traceback

# Import model classes from training script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Single_view_3D_Loss import DINOv3PoseEstimator, Research3Kinematics, IMAGE_RESOLUTION
from confidence_utils import (
    decode_keypoints_with_confidence,
    filtered_joint_summary,
    select_pnp_indices,
)

# Configuration
SYNC_CSV_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/fr3_matched_joint_angle.csv"
POSE1_ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/pose1_aruco_pose_summary.json"
POSE2_ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/pose2_aruco_pose_summary.json"
CALIB_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/franka_research3_calib_cam_from_conf"
CHECKPOINT_PATH = "/home/najo/NAS/DIP/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth"
JSON_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/franka_research3_to_DREAM"

SERIAL_TO_VIEW = {
    '41182735': "view1",
    '49429257': "view2",
    '44377151': "view3",
    '49045152': "view4"
}

def convert_to_absolute_path(relative_path):
    """Convert relative path from CSV to absolute path."""
    if relative_path.startswith('../dataset/'):
        return relative_path.replace('../dataset/', '/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/')
    elif os.path.isabs(relative_path):
        return relative_path
    else:
        return os.path.join('/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/', relative_path)

def load_model(checkpoint_path, model_type, device='cuda'):
    """Load the trained model from checkpoint."""
    # Determine model_name based on model_type, replicating logic from Single_view_3D_Loss.py
    if 'vit' in model_type:
        dino_model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    elif 'conv' in model_type:
        dino_model_name = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'
    elif 'siglip2' in model_type:
        dino_model_name = 'google/siglip2-base-patch16-224'
    elif 'siglip' in model_type:
        dino_model_name = 'google/siglip-base-patch16-224'
    else: # Default or combined
        dino_model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m' # Fallback for 'combined' or unknown

    model = DINOv3PoseEstimator(dino_model_name=dino_model_name, ablation_mode=model_type)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} with type '{model_type}'")
    return model

def preprocess_image(image_path):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0)

    return img_tensor, img_rgb

def solve_pnp_for_pose(points_3d_robot, points_2d_image, camera_matrix, dist_coeffs):
    """Solve PnP to get robot base pose (rvec, tvec) from robot coordinates to camera coordinates."""
    success, rvec, tvec = cv2.solvePnP(
        points_3d_robot.astype(np.float32),
        points_2d_image.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Warning: PnP solve failed, using identity transform")
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)

    return rvec, tvec

def transform_robot_to_camera(points_3d_robot, rvec, tvec):
    """Transform 3D points from robot coordinates to camera coordinates."""
    R_mat, _ = cv2.Rodrigues(rvec)
    points_3d_camera = (R_mat @ points_3d_robot.T).T + tvec.reshape(1, 3)
    return points_3d_camera

def visualize_3d_prediction(json_path, model, device='cuda'):
    """Visualize 3D model prediction on a single Franka Research3 sample."""
    with open(json_path, 'r') as f:
        sample = json.load(f)

    relative_image_path = sample['meta']['image_path']
    image_path = convert_to_absolute_path(relative_image_path)

    img_tensor, img_rgb = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_heatmaps, pred_angles = model(img_tensor)

    pred_kpts_2d_scaled, confidences, visibility = decode_keypoints_with_confidence(
        pred_heatmaps, img_rgb.shape
    )

    pred_angles_np = pred_angles[0].cpu().numpy()[:8]  # Research3 has 8 joint angles (7 + 1 finger)

    # Get view from metadata
    view = sample['meta']['view']

    robot = Research3Kinematics()
    joint_coords_3d_robot = robot.forward_kinematics(pred_angles_np, view=view)

    camera_matrix = np.array(sample['meta']['K'], dtype=np.float32)
    dist_coeffs = np.array(sample['meta']['dist_coeffs'], dtype=np.float32)

    selected_idx, used_fallback = select_pnp_indices(confidences, visibility, min_points=6, prefer_points=8)
    if len(selected_idx) < 4:
        pred_3d_camera = np.zeros_like(joint_coords_3d_robot)
    else:
        pred_kpts_2d_for_pnp = pred_kpts_2d_scaled[selected_idx]
        joint_coords_3d_for_pnp = joint_coords_3d_robot[selected_idx]

        # Use ArUco rvec/tvec if available, otherwise use PnP
        if 'aruco_rvec_tvec' in sample['meta']:
            rvec_data = sample['meta']['aruco_rvec_tvec']['rvec']
            tvec_data = sample['meta']['aruco_rvec_tvec']['tvec']
            rvec = np.array(rvec_data, dtype=np.float32)
            tvec = np.array(tvec_data, dtype=np.float32)
        else:
            rvec, tvec = solve_pnp_for_pose(joint_coords_3d_for_pnp, pred_kpts_2d_for_pnp, camera_matrix, dist_coeffs)

        pred_3d_camera = transform_robot_to_camera(joint_coords_3d_robot, rvec, tvec)

    # Get ground truth 3D points
    gt_3d_camera = np.array([kp['location'] for kp in sample['objects'][0]['keypoints']], dtype=np.float32)

    # Determine pose (pose1 or pose2)
    pose = "pose1" if "pose1" in image_path else "pose2"
    summary = filtered_joint_summary(confidences, visibility)
    if used_fallback:
        summary += " | PnP fallback (insufficient confident joints)"

    return gt_3d_camera, pred_3d_camera, summary, view, pose

def plot_3d_comparison(ax, gt_3d, pred_3d, title):
    """Plot 3D comparison of ground truth and predicted points."""
    # Plot ground truth (blue)
    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2],
               c='blue', marker='o', s=100, label='Ground Truth', alpha=0.6)

    # Plot connections for ground truth
    for i in range(len(gt_3d) - 1):
        ax.plot([gt_3d[i, 0], gt_3d[i+1, 0]],
                [gt_3d[i, 1], gt_3d[i+1, 1]],
                [gt_3d[i, 2], gt_3d[i+1, 2]],
                'b-', linewidth=2, alpha=0.4)

    # Plot predicted (red)
    ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2],
               c='red', marker='^', s=100, label='Predicted', alpha=0.6)

    # Plot connections for predicted
    for i in range(len(pred_3d) - 1):
        ax.plot([pred_3d[i, 0], pred_3d[i+1, 0]],
                [pred_3d[i, 1], pred_3d[i+1, 1]],
                [pred_3d[i, 2], pred_3d[i+1, 2]],
                'r-', linewidth=2, alpha=0.4)

    # Plot error lines (dashed lines connecting GT and predicted)
    for i in range(min(len(gt_3d), len(pred_3d))):
        ax.plot([gt_3d[i, 0], pred_3d[i, 0]],
                [gt_3d[i, 1], pred_3d[i, 1]],
                [gt_3d[i, 2], pred_3d[i, 2]],
                'gray', linestyle='--', linewidth=1, alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()

    # Set equal aspect ratio
    all_points = np.vstack([gt_3d, pred_3d])
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                          all_points[:, 1].max() - all_points[:, 1].min(),
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set viewing angle - base at bottom
    # elev: elevation angle (view from above when positive)
    # azim: azimuth angle (rotation around z-axis)
    ax.view_init(elev=20, azim=45)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, args.model_type, device)

    print("Franka Research 3 Dataset 3D Visualization\n" + "=" * 50)

    # Get all JSON files
    all_json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith('.json')]

    if not all_json_files:
        print(f"No JSON files found in {JSON_DIR}")
        return

    print(f"Found {len(all_json_files)} JSON files")

    # Randomly sample files and categorize by pose
    pose_samples = {'pose1': [], 'pose2': []}
    sample_size = min(200, len(all_json_files))  # Sample first to avoid reading all files
    sampled_files = random.sample(all_json_files, sample_size)

    for json_path in sampled_files:
        try:
            with open(json_path, 'r') as f:
                sample = json.load(f)
            image_path = sample['meta']['image_path']

            if 'pose1' in image_path:
                pose_samples['pose1'].append(json_path)
            elif 'pose2' in image_path:
                pose_samples['pose2'].append(json_path)

            # Stop early if we have enough samples
            if len(pose_samples['pose1']) >= 4 and len(pose_samples['pose2']) >= 4:
                break
        except Exception as e:
            continue

    # Select samples for visualization
    selected_samples = []
    for pose in ['pose1', 'pose2']:
        if pose_samples[pose]:
            num_to_select = min(4, len(pose_samples[pose]))
            selected = random.sample(pose_samples[pose], num_to_select)
            for json_path in selected:
                selected_samples.append(json_path)
                print(f"  [{pose}] Selected: {os.path.basename(json_path)}")

    if not selected_samples:
        print("No samples selected for visualization")
        return

    num_cols = 4
    num_rows = math.ceil(len(selected_samples) / num_cols)
    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, json_path in enumerate(selected_samples):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
        try:
            gt_3d, pred_3d, joint_summary, view, pose = visualize_3d_prediction(json_path, model, device)
            plot_3d_comparison(ax, gt_3d, pred_3d,
                             f"{view} ({pose})\n{os.path.basename(json_path)}")
            print(f"{os.path.basename(json_path)} -> {joint_summary}")
        except Exception as e:
            print(f"Error processing {os.path.basename(json_path)}: {e}")
            traceback.print_exc()
            ax.set_title(f"Error: {os.path.basename(json_path)}", color='red')

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Franka Research 3 robot pose predictions in 3D")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='dino_conv_only')
    args = parser.parse_args()
    main(args)
