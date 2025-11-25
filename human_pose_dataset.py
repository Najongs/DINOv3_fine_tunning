import os
import json
import numpy as np
import random
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset

# COCO Keypoint Configuration (17 keypoints)
COCO_KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

NUM_KEYPOINTS = 17


def create_gaussian_heatmap(keypoint_2d, heatmap_size, sigma):
    """
    Create a Gaussian heatmap for a single keypoint.

    Args:
        keypoint_2d: (x, y) coordinates of the keypoint
        heatmap_size: (H, W) size of the heatmap
        sigma: standard deviation for the Gaussian

    Returns:
        heatmap: (H, W) Gaussian heatmap
    """
    H, W = heatmap_size
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap


def scale_keypoints(keypoints_xy, from_size, to_size):
    """
    Scale keypoint coordinates from one image size to another.

    Args:
        keypoints_xy: (N, 2) array of keypoint coordinates
        from_size: (W, H) original image size
        to_size: (W, H) target image size

    Returns:
        scaled_keypoints: (N, 2) array of scaled keypoint coordinates
    """
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(keypoints_xy, dtype=np.float32)
    out[:, 0] = keypoints_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = keypoints_xy[:, 1] * (Ht / float(Hf))
    return out


class KeypointOcclusionAugmentor:
    """
    Randomly occludes regions around selected keypoints to simulate visibility dropouts.
    Returns both the occluded image and a per-joint visibility mask.
    """
    def __init__(
        self,
        prob=0.2,
        min_occlusions=1,
        max_occlusions=4,
        min_patch_ratio=0.05,
        max_patch_ratio=0.15,
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

    def __call__(self, img_rgb, keypoints_xy, visibilities):
        """
        Args:
            img_rgb: (H, W, 3) RGB image
            keypoints_xy: (N, 2) keypoint coordinates
            visibilities: (N,) visibility flags (0: not visible, 1: occluded, 2: visible)

        Returns:
            img_aug: augmented image
            visibilities: updated visibility flags
        """
        h, w = img_rgb.shape[:2]
        visibilities = visibilities.copy()

        if len(keypoints_xy) == 0 or random.random() > self.prob:
            return img_rgb, visibilities

        img_aug = img_rgb.copy()

        # Only occlude visible keypoints
        visible_indices = [i for i in range(len(keypoints_xy)) if visibilities[i] == 2]
        if len(visible_indices) == 0:
            return img_rgb, visibilities

        num_to_occlude = random.randint(
            self.min_occlusions,
            min(self.max_occlusions, len(visible_indices))
        )
        occluded_indices = random.sample(visible_indices, num_to_occlude)

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
            visibilities[idx] = 1  # Mark as occluded

        return img_aug, visibilities


class COCOHumanPoseDataset(Dataset):
    """
    COCO format human pose estimation dataset.

    Expected COCO annotation format:
    {
        "images": [
            {
                "id": int,
                "file_name": str,
                "width": int,
                "height": int
            }
        ],
        "annotations": [
            {
                "image_id": int,
                "keypoints": [x1, y1, v1, x2, y2, v2, ...],  # 17 keypoints * 3
                "num_keypoints": int,
                "bbox": [x, y, w, h]
            }
        ]
    }

    where vi = 0 (not labeled), 1 (labeled but not visible), 2 (labeled and visible)
    """
    def __init__(
        self,
        image_dir,
        annotation_file,
        transform,
        heatmap_size=(512, 512),
        sigma=3.0,
        occlusion_augmentor=None
    ):
        """
        Args:
            image_dir: path to image directory
            annotation_file: path to COCO format annotation JSON file
            transform: torchvision transform for images
            heatmap_size: (H, W) size of output heatmaps
            sigma: standard deviation for Gaussian heatmaps
            occlusion_augmentor: KeypointOcclusionAugmentor instance (optional)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.occlusion_augmentor = occlusion_augmentor

        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Build image_id to image info mapping
        self.images = {img['id']: img for img in coco_data['images']}

        # Build list of annotations (one per person)
        self.annotations = []
        for ann in coco_data['annotations']:
            if ann['num_keypoints'] > 0:  # Only include annotations with keypoints
                self.annotations.append(ann)

        print(f"Loaded {len(self.annotations)} annotations from {annotation_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        img_info = self.images[image_id]

        # Load image
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Failed to load image {image_path}")
            return self.__getitem__((idx + 1) % len(self))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Parse keypoints: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
        keypoints_flat = ann['keypoints']
        keypoints_xy = np.array([[keypoints_flat[i], keypoints_flat[i+1]]
                                  for i in range(0, len(keypoints_flat), 3)], dtype=np.float32)
        visibilities = np.array([keypoints_flat[i+2] for i in range(0, len(keypoints_flat), 3)],
                                dtype=np.int32)

        # Apply occlusion augmentation
        if self.occlusion_augmentor is not None:
            img_rgb, visibilities = self.occlusion_augmentor(img_rgb, keypoints_xy, visibilities)

        # Convert to PIL and apply transforms
        img_pil = Image.fromarray(img_rgb)
        image_tensor = self.transform(img_pil)

        # Create GT heatmaps
        Ht, Wt = self.heatmap_size
        kpts_on_heatmap = scale_keypoints(keypoints_xy, from_size=(w, h), to_size=(Wt, Ht))

        heatmaps_np = np.zeros((NUM_KEYPOINTS, Ht, Wt), dtype=np.float32)
        keypoint_weights = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        for i in range(NUM_KEYPOINTS):
            if visibilities[i] > 0:  # Labeled (visible or occluded)
                heatmaps_np[i] = create_gaussian_heatmap(kpts_on_heatmap[i], (Ht, Wt), self.sigma)
                keypoint_weights[i] = 1.0 if visibilities[i] == 2 else 0.5  # Lower weight for occluded
            else:  # Not labeled
                keypoint_weights[i] = 0.0

        gt_heatmaps = torch.from_numpy(heatmaps_np)  # (17, H, W)
        keypoint_weights = torch.from_numpy(keypoint_weights)  # (17,)

        return image_tensor, gt_heatmaps, keypoint_weights


def human_pose_collate_fn(batch):
    """
    Collate function for batching human pose samples.

    Args:
        batch: list of (image_tensor, gt_heatmaps, keypoint_weights)

    Returns:
        image_tensors: (B, 3, H, W)
        gt_heatmaps: (B, 17, H_heatmap, W_heatmap)
        keypoint_weights: (B, 17)
    """
    image_tensors, gt_heatmaps, keypoint_weights = zip(*batch)

    image_tensors = torch.stack(image_tensors, 0)
    gt_heatmaps = torch.stack(gt_heatmaps, 0)
    keypoint_weights = torch.stack(keypoint_weights, 0)

    return image_tensors, gt_heatmaps, keypoint_weights


if __name__ == "__main__":
    # Test the dataset
    from torchvision import transforms

    print("Testing COCOHumanPoseDataset...")

    # Example paths (adjust these to your actual COCO dataset)
    image_dir = "/path/to/coco/images/train2017"
    annotation_file = "/path/to/coco/annotations/person_keypoints_train2017.json"

    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        print("Please update the paths in __main__ section to test the dataset.")
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        occlusion_aug = KeypointOcclusionAugmentor(prob=0.2)

        dataset = COCOHumanPoseDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            transform=transform,
            heatmap_size=(512, 512),
            sigma=3.0,
            occlusion_augmentor=occlusion_aug
        )

        print(f"Dataset size: {len(dataset)}")

        # Test loading a sample
        if len(dataset) > 0:
            img, heatmaps, weights = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Heatmaps shape: {heatmaps.shape}")
            print(f"Weights shape: {weights.shape}")
            print("Dataset test passed!")
