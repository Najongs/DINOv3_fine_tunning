import os
import json
import math
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
from torchvision import transforms

# Import model classes from training script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Single_view_3D_Loss import DINOv3PoseEstimator, Research3Kinematics, get_max_preds

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

def project_to_pixel(coords_3d, rvec, tvec, camera_matrix, dist_coeffs=None):
    """Project 3D coordinates to 2D image plane."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float32)

    pixel_coords, _ = cv2.projectPoints(coords_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return pixel_coords.reshape(-1, 2)

def visualize_prediction(json_path, model, device='cuda'):
    """Visualize model prediction on a single DREAM sample."""
    with open(json_path, 'r') as f:
        sample = json.load(f)

    relative_image_path = sample['meta']['image_path']
    image_path = os.path.join(DREAM_IMAGE_BASE, relative_image_path.replace('../dataset/DREAM_real/', ''))

    img_tensor, img_rgb = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_heatmaps, pred_angles = model(img_tensor)

    pred_heatmaps_np = pred_heatmaps.cpu().numpy()
    pred_kpts_2d, confidences = get_max_preds(pred_heatmaps_np)
    pred_kpts_2d = pred_kpts_2d[0]

    h, w = img_rgb.shape[:2]
    heatmap_size = (224, 224)
    pred_kpts_2d_scaled = pred_kpts_2d * np.array([w, h]) / np.array(heatmap_size)

    pred_angles_np = pred_angles[0].cpu().numpy()[:8]

    robot = Research3Kinematics()
    joint_coords_3d_robot = robot.forward_kinematics(pred_angles_np, view='view1')

    camera_matrix = np.array(sample['meta']['K'], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    undistorted_img = img_rgb.copy()

    num_keypoints = min(8, len(pred_kpts_2d_scaled), len(joint_coords_3d_robot))
    if num_keypoints < 4:
        pixel_coords_fk = None
    else:
        pred_kpts_2d_for_pnp = pred_kpts_2d_scaled[:num_keypoints]
        joint_coords_3d_for_pnp = joint_coords_3d_robot[:num_keypoints]
        rvec, tvec = solve_pnp_for_pose(joint_coords_3d_for_pnp, pred_kpts_2d_for_pnp, camera_matrix, dist_coeffs)
        pixel_coords_fk = project_to_pixel(joint_coords_3d_robot, rvec, tvec, camera_matrix, dist_coeffs)

    gt_kpts_2d = np.array([kp['projected_location'] for kp in sample['objects'][0]['keypoints']], dtype=np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (x, y) in enumerate(gt_kpts_2d.astype(int)):
        cv2.circle(undistorted_img, (x, y), 4, (0, 255, 255), -1)
        if idx > 0: cv2.line(undistorted_img, tuple(gt_kpts_2d[idx-1].astype(int)), (x, y), (0, 255, 255), 1)

    num_heatmap_keypoints = min(8, len(pred_kpts_2d_scaled))
    for idx in range(num_heatmap_keypoints):
        x, y = pred_kpts_2d_scaled[idx].astype(int)
        cv2.circle(undistorted_img, (x, y), 6, (0, 255, 0), -1)
        if idx > 0: cv2.line(undistorted_img, tuple(pred_kpts_2d_scaled[idx-1].astype(int)), (x, y), (0, 255, 0), 2)

    if pixel_coords_fk is not None:
        for idx, (x, y) in enumerate(pixel_coords_fk.astype(int)):
            cv2.circle(undistorted_img, (x, y), 8, (255, 0, 255), -1)
            if idx > 0: cv2.line(undistorted_img, tuple(pixel_coords_fk[idx-1].astype(int)), (x, y), (255, 0, 255), 3)

    return undistorted_img

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, args.model_type, device)

    print("DREAM Dataset Visualization\n" + "=" * 50)
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
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
    axes = axes.flatten() if len(selected_samples) > 1 else [axes]

    for i, (dataset_name, json_path) in enumerate(selected_samples):
        ax = axes[i]
        try:
            result_img = visualize_prediction(json_path, model, device)
            ax.imshow(result_img)
            ax.set_title(f"{dataset_name}\n{os.path.basename(json_path)}", fontsize=12)
        except Exception as e:
            print(f"Error processing {dataset_name}/{os.path.basename(json_path)}: {e}")
            traceback.print_exc()
            ax.set_title(f"Error: {dataset_name}", color='red')
        finally:
            ax.axis("off")

    for j in range(len(selected_samples), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize DREAM dataset robot pose predictions")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='dino_conv_only')
    args = parser.parse_args()
    main(args)
