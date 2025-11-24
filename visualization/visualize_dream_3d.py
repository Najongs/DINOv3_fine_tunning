import os
import json
import math
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torchvision import transforms

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
DREAM_JSON_BASE = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM"
DREAM_IMAGE_BASE = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real"
CHECKPOINT_PATH = "/home/najo/NAS/DIP/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth"

DREAM_DATASETS = [
    "panda-3cam_azure",
    "panda-3cam_kinect360",
    "panda-3cam_realsense",
    "panda-orb"
]

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

    # Handle size mismatch for keypoint head (7 vs 8 keypoints)
    # Load checkpoint weights, skipping mismatched layers
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in new_state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                # Skip layers with size mismatch (e.g., keypoint_head with different num_joints)
                print(f"  Skipping {k} due to size mismatch: {v.shape} vs {model_dict[k].shape}")
        else:
            print(f"  Skipping {k}: not in current model")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} ({len(filtered_dict)}/{len(new_state_dict)} layers) with type '{model_type}'")
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

def solve_pnp_for_pose(points_3d_robot, points_2d_image, camera_matrix, dist_coeffs=None):
    """
    Solve PnP to get robot base pose (rvec, tvec) from robot coordinates to camera coordinates.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

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
    """Visualize 3D model prediction on a single DREAM sample."""
    with open(json_path, 'r') as f:
        sample = json.load(f)

    relative_image_path = sample['meta']['image_path']
    image_path = os.path.join(DREAM_IMAGE_BASE, relative_image_path.replace('../dataset/DREAM_real/', ''))

    img_tensor, img_rgb = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_heatmaps, pred_angles = model(img_tensor)

    pred_kpts_2d_scaled, confidences, visibility = decode_keypoints_with_confidence(
        pred_heatmaps, img_rgb.shape
    )

    pred_angles_np = pred_angles[0].cpu().numpy()[:8]

    robot = Research3Kinematics()
    joint_coords_3d_robot = robot.forward_kinematics(pred_angles_np, view='view1')

    camera_matrix = np.array(sample['meta']['K'], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    selected_idx, used_fallback = select_pnp_indices(confidences, visibility, min_points=6, prefer_points=8)
    if len(selected_idx) < 4:
        pred_3d_camera = np.zeros_like(joint_coords_3d_robot)
    else:
        pred_kpts_2d_for_pnp = pred_kpts_2d_scaled[selected_idx]
        joint_coords_3d_for_pnp = joint_coords_3d_robot[selected_idx]
        rvec, tvec = solve_pnp_for_pose(joint_coords_3d_for_pnp, pred_kpts_2d_for_pnp, camera_matrix, dist_coeffs)
        pred_3d_camera = transform_robot_to_camera(joint_coords_3d_robot, rvec, tvec)

    # Get ground truth 3D points
    gt_3d_camera = np.array([kp['location'] for kp in sample['objects'][0]['keypoints']], dtype=np.float32)
    summary = filtered_joint_summary(confidences, visibility)
    if used_fallback:
        summary += " | PnP fallback (insufficient confident joints)"

    return gt_3d_camera, pred_3d_camera, summary

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

    print("DREAM Dataset 3D Visualization\n" + "=" * 50)
    selected_samples = []
    for dataset_name in DREAM_DATASETS:
        json_dir = os.path.join(DREAM_JSON_BASE, dataset_name)
        if not os.path.isdir(json_dir): continue
        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        if len(json_files) < 2: continue
        selected = random.sample(json_files, min(2, len(json_files)))
        for json_file in selected:
            selected_samples.append((dataset_name, os.path.join(json_dir, json_file)))
            print(f"  [{dataset_name}] Selected: {json_file}")

    if not selected_samples:
        print("No samples selected for visualization")
        return

    num_cols = 4
    num_rows = math.ceil(len(selected_samples) / num_cols)
    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, (dataset_name, json_path) in enumerate(selected_samples):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')
        try:
            gt_3d, pred_3d, joint_summary = visualize_3d_prediction(json_path, model, device)
            plot_3d_comparison(ax, gt_3d, pred_3d,
                             f"{dataset_name}\n{os.path.basename(json_path)}")
            print(f"{dataset_name}/{os.path.basename(json_path)} -> {joint_summary}")
        except Exception as e:
            print(f"Error processing {dataset_name}/{os.path.basename(json_path)}: {e}")
            import traceback
            traceback.print_exc()
            ax.set_title(f"Error: {dataset_name}", color='red')

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize DREAM dataset robot pose predictions in 3D")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='dino_conv_only')
    args = parser.parse_args()
    main(args)
