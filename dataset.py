import os
import json
import numpy as np
import random
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

# Dataset Root
DATASET_ROOT = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
DREAM_DATASET_PATH = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM"
IMAGE_RESOLUTION = (1280, 720)
HEATMAP_SIZE = (1280, 720)


def create_gt_heatmap(keypoint_2d, heatmap_size, sigma):
    H, W = heatmap_size
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

def _scale_points(points_xy, from_size, to_size):
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = points_xy[:, 1] * (Ht / float(Hf))
    return out

class KeypointOcclusionAugmentor:
    """
    Randomly occludes regions around selected keypoints to simulate visibility dropouts.
    Returns both the occluded image and a per-joint confidence mask that can be used
    as a confidence-based weight inside the loss.
    """
    def __init__(
        self,
        prob=0.4,
        min_occlusions=1,
        max_occlusions=3,
        min_patch_ratio=0.05,
        max_patch_ratio=0.18,
        center_jitter=0.05,
        occluded_confidence=0.1,
        fill_with_noise=True,
    ):
        self.prob = prob
        self.min_occlusions = min_occlusions
        self.max_occlusions = max_occlusions
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio
        self.center_jitter = center_jitter
        self.occluded_confidence = occluded_confidence
        self.fill_with_noise = fill_with_noise

    def __call__(self, img_rgb, keypoints_xy):
        h, w = img_rgb.shape[:2]
        joint_conf = np.ones(len(keypoints_xy), dtype=np.float32)

        if len(keypoints_xy) == 0 or random.random() > self.prob:
            return img_rgb, joint_conf

        img_aug = img_rgb.copy()
        num_to_occlude = random.randint(
            self.min_occlusions, min(self.max_occlusions, len(keypoints_xy))
        )
        occluded_indices = random.sample(range(len(keypoints_xy)), num_to_occlude)

        for idx in occluded_indices:
            cx, cy = keypoints_xy[idx]
            cx = np.clip(
                cx + random.uniform(-self.center_jitter, self.center_jitter) * w,
                0,
                w - 1,
            )
            cy = np.clip(
                cy + random.uniform(-self.center_jitter, self.center_jitter) * h,
                0,
                h - 1,
            )
            patch_w = int(
                max(1, random.uniform(self.min_patch_ratio, self.max_patch_ratio) * w)
            )
            patch_h = int(
                max(1, random.uniform(self.min_patch_ratio, self.max_patch_ratio) * h)
            )

            left = int(max(0, cx - patch_w / 2))
            right = int(min(w, cx + patch_w / 2))
            top = int(max(0, cy - patch_h / 2))
            bottom = int(min(h, cy + patch_h / 2))

            if right <= left or bottom <= top:
                continue

            if self.fill_with_noise:
                patch = np.random.randint(
                    0, 256, size=(bottom - top, right - left, 3), dtype=np.uint8
                )
            else:
                color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
                patch = np.tile(color, (bottom - top, right - left, 1))

            img_aug[top:bottom, left:right] = patch
            joint_conf[idx] = self.occluded_confidence

        return img_aug, joint_conf

class RobotPoseDataset(Dataset):
    def __init__(self, json_files, transform, sigma=2.0, occlusion_augmentor=None):
        self.json_files = json_files
        self.transform = transform
        self.sigma = sigma
        self.occlusion_augmentor = occlusion_augmentor

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        
        relative_image_path = sample['meta']['image_path']
        image_path = os.path.join(DATASET_ROOT, relative_image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # 다음 샘플로 대체
            return self.__getitem__((idx + 1) % len(self))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        # Per-joint confidence defaults to 1.0 (fully visible)
        keypoints = sample["objects"][0]["keypoints"]
        joint_num = len(keypoints)
        joint_confidences = np.ones(joint_num, dtype=np.float32)
        if self.occlusion_augmentor is not None:
            img_rgb, joint_confidences = self.occlusion_augmentor(
                img_rgb, [kp["projected_location"] for kp in keypoints]
            )
        img_pil = Image.fromarray(img_rgb)
        image_tensor = self.transform(img_pil)

        # 2D heatmap GT
        Ht, Wt = HEATMAP_SIZE
        kpts_2d_orig = np.array([kp["projected_location"] for kp in keypoints], dtype=np.float32)
        kpts_on_heatmap = _scale_points(kpts_2d_orig, from_size=(w, h), to_size=(Wt, Ht))
        heatmaps_np = np.zeros((joint_num, Ht, Wt), dtype=np.float32)
        for i in range(joint_num):
            heatmaps_np[i] = create_gt_heatmap(kpts_on_heatmap[i], (Ht, Wt), self.sigma)
        gt_heatmaps = torch.from_numpy(heatmaps_np)  # (J, H, W)

        
        angles = [angle["position"] for angle in sample['sim_state']["joints"]]
        gt_angles = torch.tensor(angles, dtype=torch.float32)  # (A,)
        gt_class = sample['objects'][0]['class'] # Get robot class early
        gt_3d_points = torch.tensor([kp["location"] for kp in keypoints])  # (J,3)
        
        try:
            K = torch.tensor(sample['meta']['K'], dtype=torch.float32)  # (3,3)
        except KeyError as e:
            print(f"Error: Key '{e}' (camera intrinsic K matrix) not found in sample['meta'] for {json_path}. This is critical for PnP. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        if json_path.startswith(DREAM_DATASET_PATH):
            dist = torch.zeros(5, dtype=torch.float32)
        else:
            try:
                dist = torch.tensor(sample['meta']['dist_coeffs'], dtype=torch.float32)
            except KeyError:
                print(f"Error: 'dist_coeffs' not found for non-DREAM dataset sample: {json_path}. Skipping sample.")
                return self.__getitem__((idx + 1) % len(self))
        
        orig_img_size = torch.tensor([w, h], dtype=torch.float32)
        joint_confidences = torch.from_numpy(joint_confidences)
            
        return image_tensor, gt_heatmaps, gt_angles, gt_class, gt_3d_points, K, dist, orig_img_size, joint_confidences

def robot_collate_fn(batch):
    (image_tensor, gt_heatmaps, gt_angles, gt_class,
     gt_3d_points, K, dist, orig_img_sizes, joint_confidences) = zip(*batch)

    image_tensors = torch.stack(image_tensor, 0)

    MAX_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.
    MAX_ANGLES = 9
    MAX_POINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.
    MAX_DIST_COEFFS = 14  # OpenCV supports up to 14 distortion coefficients

    heatmaps_padded = torch.zeros(len(gt_heatmaps), MAX_JOINTS, gt_heatmaps[0].shape[1], gt_heatmaps[0].shape[2])
    angles_padded   = torch.zeros(len(gt_angles), MAX_ANGLES)
    points_padded   = torch.zeros(len(gt_3d_points), MAX_POINTS, 3)
    dist_padded     = torch.zeros(len(dist), MAX_DIST_COEFFS)
    joint_conf_padded = torch.ones(len(joint_confidences), MAX_JOINTS)

    joint_lengths = torch.zeros(len(gt_heatmaps), dtype=torch.long)  # 각 샘플의 joint 개수
    angle_lengths = torch.zeros(len(gt_angles), dtype=torch.long)    # 각 샘플의 angle 개수
    point_lengths = torch.zeros(len(gt_3d_points), dtype=torch.long) # 각 샘플의 3D point 개수

    for i, (h, a, p, d, c) in enumerate(zip(gt_heatmaps, gt_angles, gt_3d_points, dist, joint_confidences)):
        joint_num = h.shape[0]
        angle_num = a.shape[0]
        point_num = p.shape[0]
        dist_num = d.shape[0]
        conf_num = c.shape[0]

        heatmaps_padded[i, :joint_num, :, :] = h
        angles_padded[i, :angle_num] = a
        points_padded[i, :point_num] = p
        dist_padded[i, :dist_num] = d
        joint_conf_padded[i, :conf_num] = c

        joint_lengths[i] = joint_num
        angle_lengths[i] = angle_num
        point_lengths[i] = point_num

    K = torch.stack(K, 0)
    orig_img_sizes = torch.stack(orig_img_sizes, 0)

    return (image_tensors, heatmaps_padded, angles_padded, gt_class, points_padded,
            K, dist_padded, joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_conf_padded)
