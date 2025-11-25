import os
import json
import math
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torchvision import transforms

# Import model classes from training script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DINOv3PoseEstimator
from kinematics import MecaInsertionKinematics
from dataset import IMAGE_RESOLUTION, HEATMAP_SIZE
from confidence_utils import (
    decode_keypoints_with_confidence,
    annotate_confidence_panel,
    filtered_joint_summary,
)

# Configuration
SYNC_CSV_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca_insertion/Meca_insertion_matched_joint_angle.csv"
ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca_insertion/Meca_insertion_aruco_pose_summary.json"
CALIB_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca_insertion/Meca_calib_cam_from_conf"
CHECKPOINT_PATH = "/home/najo/NAS/DIP/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth"

CAMERA_SERIALS = {
    "right": '49429257',
    "left": '44377151',
    "top": '49045152'
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

    model = DINOv3PoseEstimator(dino_model_name=dino_model_name, heatmap_size=HEATMAP_SIZE, ablation_mode=model_type)

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

def project_to_pixel(coords_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D coordinates to 2D image plane."""
    pixel_coords, _ = cv2.projectPoints(coords_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return pixel_coords.reshape(-1, 2)

def visualize_prediction(image_path, model, aruco_result, calib_data, device='cuda'):
    """Visualize model prediction on a single image."""
    # Preprocess image
    img_tensor, img_rgb = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Model inference
    with torch.no_grad():
        pred_heatmaps, pred_angles = model(img_tensor)

    pred_kpts_2d_scaled, confidences, visibility = decode_keypoints_with_confidence(
        pred_heatmaps, img_rgb.shape
    )

    # Get predicted angles (MecaInsertion has 6 joints)
    pred_angles_np = pred_angles[0].cpu().numpy()[:6]

    # Compute FK to get 3D joint positions
    robot = MecaInsertionKinematics()
    joint_coords_3d = robot.forward_kinematics(pred_angles_np)

    # Load camera calibration
    camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

    # Undistort image
    undistorted_img = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)

    # Get ArUco transformation
    rvec_deg = np.array([
        aruco_result.get('rvec_x_deg', aruco_result.get('rvec_x', 0)),
        aruco_result.get('rvec_y_deg', aruco_result.get('rvec_y', 0)),
        aruco_result.get('rvec_z_deg', aruco_result.get('rvec_z', 0))
    ], dtype=np.float32)
    rvec = np.deg2rad(rvec_deg)

    tvec = np.array([
        aruco_result['tvec_x'],
        aruco_result['tvec_y'],
        aruco_result['tvec_z']
    ], dtype=np.float32).reshape(3, 1)

    # Project 3D FK points to 2D
    pixel_coords_fk = project_to_pixel(joint_coords_3d, rvec, tvec, camera_matrix, np.zeros_like(dist_coeffs))

    # Draw both predictions: heatmap-based (green) and FK-based (magenta)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw heatmap-based predictions (green if kept, red if filtered)
    prev_visible_idx = None
    for idx, (point, conf, vis) in enumerate(zip(pred_kpts_2d_scaled, confidences, visibility)):
        if idx >= len(joint_coords_3d):
            break
        x, y = point.astype(int)
        label = f"H{idx}:{conf:.2f}"
        if vis:
            cv2.circle(undistorted_img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(undistorted_img, label, (x + 10, y - 10), font, 0.5, (0, 255, 0), 1)
            if prev_visible_idx is not None:
                px, py = pred_kpts_2d_scaled[prev_visible_idx].astype(int)
                cv2.line(undistorted_img, (px, py), (x, y), (0, 255, 0), 2)
            prev_visible_idx = idx
        else:
            cv2.circle(undistorted_img, (x, y), 6, (0, 0, 255), 1)
            cv2.line(undistorted_img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 1)
            cv2.line(undistorted_img, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 1)
            cv2.putText(undistorted_img, f"{label}-DROP", (x + 10, y - 10), font, 0.45, (0, 0, 255), 1)

    # Draw FK-based predictions (magenta)
    for idx, (x, y) in enumerate(pixel_coords_fk.astype(int)):
        cv2.circle(undistorted_img, (x, y), 8, (255, 0, 255), -1)
        cv2.putText(undistorted_img, f"J{idx}", (x + 10, y + 10), font, 0.5, (255, 0, 255), 1)
        if idx > 0:
            prev_x, prev_y = pixel_coords_fk[idx-1].astype(int)
            cv2.line(undistorted_img, (prev_x, prev_y), (x, y), (255, 0, 255), 3)

    undistorted_img = annotate_confidence_panel(undistorted_img, confidences, visibility)
    summary = filtered_joint_summary(confidences, visibility)
    return undistorted_img, summary

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.model_type, device)

    # Load data
    df_sync = pd.read_csv(SYNC_CSV_PATH)
    with open(ARUCO_JSON_PATH, 'r') as f:
        aruco_results = json.load(f)

    print("Data loaded successfully")

    # Select random images for each view/camera combination
    views_to_plot = ['left', 'right', 'top']
    cams_to_plot = ['leftcam', 'rightcam']
    selected_rows = []

    print("--- Selecting random representative images ---")
    for view in views_to_plot:
        for cam in cams_to_plot:
            cam_short_name = cam.replace("cam", "")
            matching_df = df_sync[
                df_sync['image_path'].str.contains(f"/{view}/") &
                df_sync['image_path'].str.contains(f"_{cam_short_name}_")
            ]

            if not matching_df.empty:
                selected_row = matching_df.sample(n=1).iloc[0]
                selected_rows.append(selected_row)
                img_name = os.path.basename(selected_row['image_path'])
                print(f"  [{view}/{cam}] Selected: {img_name}")
            else:
                print(f"  [{view}/{cam}] No matching data found")

    if not selected_rows:
        print("No data selected for visualization")
        return

    # Visualize selected data
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()

    for i, row in enumerate(selected_rows):
        ax = axes[i]
        view, cam = "N/A", "N/A"
        try:
            image_path = convert_to_absolute_path(row['image_path'])

            # Parse view and cam from image path
            img_name = os.path.basename(image_path)
            parts = os.path.splitext(img_name)[0].split('_')
            view = [v for v in views_to_plot if f"/{v}/" in image_path][0]
            cam = parts[-2] + "cam"

            # Find matching ArUco result
            aruco_result = next(item for item in aruco_results
                                if item['view'] == view and item['cam'] == cam)

            # Load calibration
            serial = CAMERA_SERIALS[view]
            calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)

            # Visualize prediction
            result_img, joint_summary = visualize_prediction(image_path, model, aruco_result, calib_data, device)

            ax.imshow(result_img)
            ax.set_title(f"View: {view.upper()} / Cam: {cam.upper()}", fontsize=14)
            print(f"{os.path.basename(image_path)} -> {joint_summary}")

        except Exception as e:
            print(f"Error processing [{view}/{cam}]: {e}")
            ax.set_title(f"Error: {view}/{cam}", color='red')
        finally:
            ax.axis("off")

    # Disable unused subplots
    for j in range(len(selected_rows), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Meca Insertion robot pose predictions")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path to save visualization (if not specified, will display)')
    parser.add_argument('--model_type', type=str, default='dino_conv_only',
                        help='DINOv3 model type (e.g., dino_conv_only, combined, siglip_only)')
    args = parser.parse_args()

    main(args)
