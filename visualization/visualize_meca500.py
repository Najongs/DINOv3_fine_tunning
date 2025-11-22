import os
import json
import math
import cv2
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
from Single_view_3D_Loss import DINOv3PoseEstimator, Meca500Kinematics, get_max_preds

# Configuration
SYNC_CSV_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca500/Meca500_matched_joint_angle.csv"
ARUCO_JSON_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca500/Meca500_aruco_pose_summary.json"
CALIB_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Meca500/Meca500_calib_cam_from_conf/Meca500_calib.json"
CHECKPOINT_PATH = "/home/najo/NAS/DIP/DINOv3_fine_tunning/checkpoints_total_dino_conv_only/best_model.pth"

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

    # Get predicted angles (Meca500 has 6 joints)
    pred_angles_np = pred_angles[0].cpu().numpy()[:6]

    # Compute FK to get 3D joint positions
    robot = Meca500Kinematics()
    joint_coords_3d = robot.forward_kinematics(pred_angles_np)

    # Load camera calibration
    camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float32)

    # Undistort image
    undistorted_img = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)

    # Get ArUco transformation
    rvec_deg = np.array([
        aruco_result.get('rvec_x', 0),
        aruco_result.get('rvec_y', 0),
        aruco_result.get('rvec_z', 0)
    ], dtype=np.float32)
    rvec = np.deg2rad(rvec_deg)

    tvec = np.array([
        aruco_result.get('tvec_x', 0),
        aruco_result.get('tvec_y', 0),
        aruco_result.get('tvec_z', 0)
    ], dtype=np.float32).reshape(3, 1)

    # Project 3D FK points to 2D
    pixel_coords_fk = project_to_pixel(joint_coords_3d, rvec, tvec, camera_matrix, np.zeros_like(dist_coeffs))

    # Draw both predictions: heatmap-based (green) and FK-based (magenta)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw heatmap-based predictions (green)
    for idx, (x, y) in enumerate(pred_kpts_2d_scaled.astype(int)):
        if idx >= len(joint_coords_3d):
            break
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
    with open(ARUCO_JSON_PATH, 'r') as f:
        aruco_results = json.load(f)
    with open(CALIB_PATH, 'r') as f:
        calib_data = json.load(f)

    # Single camera environment, use first ArUco result
    if not aruco_results:
        print("Error: ArUco JSON is empty")
        return
    aruco_result = aruco_results[0]

    print("Data loaded successfully")

    # Select random samples for visualization
    num_samples = args.num_samples
    if len(df_sync) < num_samples:
        print(f"Warning: Dataset has {len(df_sync)} samples, fewer than requested {num_samples}")
        num_samples = len(df_sync)

    if df_sync.empty:
        print("Error: No data in CSV file")
        return

    selected_rows = df_sync.sample(n=num_samples).to_dict('records')
    print(f"Selected {len(selected_rows)} random images for visualization")

    # Create subplot grid
    num_cols = 3
    num_rows = math.ceil(len(selected_rows) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 8, num_rows * 6))
    axes = axes.flatten()

    for i, row in enumerate(selected_rows):
        ax = axes[i]
        try:
            image_path = convert_to_absolute_path(row['image_path'])

            # Visualize prediction
            result_img = visualize_prediction(image_path, model, aruco_result, calib_data, device)

            ax.imshow(result_img)
            ax.set_title(os.path.basename(image_path), fontsize=12)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            ax.set_title(f"Error", color='red')
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
    parser = argparse.ArgumentParser(description="Visualize Meca500 robot pose predictions")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of random samples to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path to save visualization (if not specified, will display)')
    parser.add_argument('--model_type', type=str, default='dino_conv_only',
                        help='DINOv3 model type (e.g., dino_conv_only, combined, siglip_only)')
    args = parser.parse_args()

    main(args)
