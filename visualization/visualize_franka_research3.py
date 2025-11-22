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
import traceback

# Import model classes from training script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Single_view_3D_Loss import DINOv3PoseEstimator, Research3Kinematics, get_max_preds

# Configuration
SYNC_CSV_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/fr3_matched_joint_angle.csv"
POSE1_ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/pose1_aruco_pose_summary.json"
POSE2_ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/pose2_aruco_pose_summary.json"
CALIB_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/franka_research3/franka_research3_calib_cam_from_conf"
CHECKPOINT_PATH = "/home/najo/NAS/DIP/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth"

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
        transforms.Resize((512, 512)),
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

def project_to_pixel(coords_3d, aruco_result, camera_matrix, dist_coeffs):
    """Project 3D coordinates to 2D image plane."""
    rvec = np.array([
        aruco_result['rvec_x'],
        aruco_result['rvec_y'],
        aruco_result['rvec_z']
    ], dtype=np.float32)
    tvec = np.array([
        aruco_result['tvec_x'],
        aruco_result['tvec_y'],
        aruco_result['tvec_z']
    ], dtype=np.float32).reshape(3, 1)

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

    # Extract 2D keypoints from heatmaps
    pred_heatmaps_np = pred_heatmaps.cpu().numpy()
    pred_kpts_2d, confidences = get_max_preds(pred_heatmaps_np)
    pred_kpts_2d = pred_kpts_2d[0]  # First batch item

    # Scale keypoints from heatmap size (224x224) to original image size
    h, w = img_rgb.shape[:2]
    heatmap_size = (224, 224)
    scale_x = w / heatmap_size[0]
    scale_y = h / heatmap_size[1]
    pred_kpts_2d_scaled = pred_kpts_2d.copy()
    pred_kpts_2d_scaled[:, 0] *= scale_x
    pred_kpts_2d_scaled[:, 1] *= scale_y

    # Get predicted angles (Research3 has 7 joints + 1 finger)
    pred_angles_np = pred_angles[0].cpu().numpy()[:8]  # Use first 8 joint angles

    # Get view from aruco_result
    view = aruco_result['view']

    # Compute FK to get 3D joint positions
    robot = Research3Kinematics()
    joint_coords_3d = robot.forward_kinematics(pred_angles_np, view=view)

    # Load camera calibration
    camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

    # Undistort image
    undistorted_img = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)

    # Project 3D FK points to 2D
    pixel_coords_fk = project_to_pixel(joint_coords_3d, aruco_result, camera_matrix, np.zeros_like(dist_coeffs))

    # Draw both predictions: heatmap-based (green) and FK-based (magenta)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw heatmap-based predictions (green) - show all 8 keypoints (7 joints + 1 finger)
    num_heatmap_keypoints = min(8, len(pred_kpts_2d_scaled))  # Research3 has 8 keypoints
    for idx in range(num_heatmap_keypoints):
        x, y = pred_kpts_2d_scaled[idx].astype(int)
        cv2.circle(undistorted_img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(undistorted_img, f"H{idx}", (x + 10, y - 10), font, 0.5, (0, 255, 0), 1)
        if idx > 0:
            prev_x, prev_y = pred_kpts_2d_scaled[idx-1].astype(int)
            cv2.line(undistorted_img, (prev_x, prev_y), (x, y), (0, 255, 0), 2)

    # Draw FK-based predictions (magenta)
    for idx, (x, y) in enumerate(pixel_coords_fk.astype(int)):
        cv2.circle(undistorted_img, (x, y), 8, (255, 0, 255), -1)
        cv2.putText(undistorted_img, f"J{idx}", (x + 10, y + 10), font, 0.5, (255, 0, 255), 1)
        if idx > 0:
            prev_x, prev_y = pixel_coords_fk[idx-1].astype(int)
            cv2.line(undistorted_img, (prev_x, prev_y), (x, y), (255, 0, 255), 3)

    return undistorted_img

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.model_type, device)

    # Load data
    df_sync = pd.read_csv(SYNC_CSV_PATH)
    with open(POSE1_ARUCO_JSON_PATH, 'r') as f:
        pose1_aruco_results = json.load(f)
    with open(POSE2_ARUCO_JSON_PATH, 'r') as f:
        pose2_aruco_results = json.load(f)

    print("Data loaded successfully")
    print(f"CSV columns: {df_sync.columns.tolist()}")

    # Select multi-view data from the same timestamp for each pose
    poses_to_plot = ['pose1', 'pose2']
    selected_data = []

    print("\n--- Selecting multi-view data from same timestamp for each pose ---")
    for pose in poses_to_plot:
        # Filter data for this pose
        df_pose = df_sync[df_sync['image_path'].str.contains(pose)]
        if df_pose.empty:
            print(f"  No data found for '{pose}'")
            continue

        # Count views per timestamp
        timestamp_counts = df_pose['robot_timestamp'].value_counts()
        if timestamp_counts.empty:
            print(f"  No valid timestamps for '{pose}'")
            continue

        # Find timestamp with most views
        max_views = timestamp_counts.max()
        best_timestamps = timestamp_counts[timestamp_counts == max_views].index

        if len(best_timestamps) == 0:
            print(f"  Could not find multi-view data for '{pose}'")
            continue

        # Randomly select one of the best timestamps
        target_timestamp = random.choice(best_timestamps)

        # Get all rows with this timestamp
        time_matched_rows = df_pose[df_pose['robot_timestamp'] == target_timestamp]

        for _, row in time_matched_rows.iterrows():
            selected_data.append(row)

        print(f"  '{pose}': Selected {len(time_matched_rows)} views from timestamp {target_timestamp:.4f}")

    if not selected_data:
        print("\nNo data selected for visualization")
        return

    # Create subplot grid
    num_plots = len(selected_data)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, row in enumerate(selected_data):
        image_path = convert_to_absolute_path(row['image_path'])

        try:
            # Parse image filename
            img_name = os.path.basename(image_path)
            parts = img_name.split('_')
            serial_str = parts[1]
            cam_str = parts[2]
            view_str = SERIAL_TO_VIEW[serial_str]

            # Determine which ArUco data to use
            if 'pose1' in image_path:
                current_aruco_data = pose1_aruco_results
                pose_name = "pose1"
            elif 'pose2' in image_path:
                current_aruco_data = pose2_aruco_results
                pose_name = "pose2"
            else:
                axes[i].set_title(f"[{view_str}/{cam_str}]\nPose info missing", color='orange')
                axes[i].axis("off")
                continue

            # Find matching ArUco result
            aruco_result = next(item for item in current_aruco_data
                                if item['view'] == view_str and item['cam'] == (cam_str + 'cam'))

            # Load calibration
            calib_path = os.path.join(CALIB_DIR, f"{view_str}_{serial_str}_{cam_str + 'cam'}_calib.json")
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)

            # Visualize prediction
            result_img = visualize_prediction(image_path, model, aruco_result, calib_data, device)

            axes[i].imshow(result_img)
            axes[i].set_title(f"{view_str} / {cam_str}\n({pose_name})", fontsize=12)
            axes[i].axis("off")

        except StopIteration:
            axes[i].set_title(f"[{view_str}/{cam_str}]\nArUco Data Not Found", color='red')
            axes[i].axis("off")
        except FileNotFoundError as e:
            axes[i].set_title(f"[{view_str}/{cam_str}]\nCalib File Not Found", color='red')
            axes[i].axis("off")
        except Exception as e:
            print(f"\nError processing {image_path}:")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error Details: {e}")
            traceback.print_exc()
            axes[i].set_title(f"ERROR", color='purple')
            axes[i].axis("off")

    # Disable unused subplots
    for j in range(len(selected_data), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=2.0)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Franka Research 3 robot pose predictions")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path to save visualization (if not specified, will display)')
    parser.add_argument('--model_type', type=str, default='dino_conv_only',
                        help='DINOv3 model type (e.g., dino_conv_only, combined, siglip_only)')
    args = parser.parse_args()

    main(args)
