"""
Depth dataset loader for training DINOv3DepthEstimator.
Loads RGB images and corresponding depth ground truth from teacher model (Depth Anything 3).
"""
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    """
    Dataset for monocular depth estimation training.
    Loads RGB images and corresponding teacher depth maps.
    """
    def __init__(self, rgb_root, depth_root, transform=None, extensions=['.jpg', '.jpeg', '.png']):
        """
        Args:
            rgb_root: Root directory of RGB images
            depth_root: Root directory of depth .npy files
            transform: Transform to apply to RGB images
            extensions: Image file extensions to look for
        """
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(depth_root)
        self.transform = transform

        # Find all depth files
        self.depth_files = []
        for ext in ['.npy']:
            self.depth_files.extend(list(self.depth_root.rglob(f'*{ext}')))

        # Match with RGB images
        self.samples = []
        for depth_path in self.depth_files:
            # Get relative path
            rel_path = depth_path.relative_to(self.depth_root)

            # Find corresponding RGB image
            rgb_found = False
            for img_ext in extensions:
                rgb_rel_path = rel_path.with_suffix(img_ext)
                rgb_path = self.rgb_root / rgb_rel_path
                if rgb_path.exists():
                    self.samples.append((str(rgb_path), str(depth_path)))
                    rgb_found = True
                    break

            if not rgb_found:
                # Try uppercase extensions
                for img_ext in [e.upper() for e in extensions]:
                    rgb_rel_path = rel_path.with_suffix(img_ext)
                    rgb_path = self.rgb_root / rgb_rel_path
                    if rgb_path.exists():
                        self.samples.append((str(rgb_path), str(depth_path)))
                        break

        print(f"Found {len(self.depth_files)} depth files")
        print(f"Matched {len(self.samples)} RGB-Depth pairs")

        if len(self.samples) == 0:
            raise RuntimeError(f"No RGB-Depth pairs found! Check paths:\n"
                             f"  RGB root: {rgb_root}\n"
                             f"  Depth root: {depth_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        # Load RGB image
        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            # Try next sample if image loading fails
            return self.__getitem__((idx + 1) % len(self))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Apply transform to RGB
        if self.transform is not None:
            image_tensor = self.transform(img_pil)
        else:
            image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Load depth GT
        try:
            depth_gt = np.load(depth_path)  # (H, W)
            depth_tensor = torch.from_numpy(depth_gt).unsqueeze(0).float()  # (1, H, W)
        except Exception as e:
            print(f"Error loading depth: {depth_path}, {e}")
            return self.__getitem__((idx + 1) % len(self))

        return image_tensor, depth_tensor


def depth_collate_fn(batch):
    """
    Custom collate function for depth dataset.

    Args:
        batch: List of (image_tensor, depth_tensor) tuples

    Returns:
        images: (B, 3, H, W)
        depths: (B, 1, H_d, W_d)
    """
    images, depths = zip(*batch)

    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)

    return images, depths


if __name__ == "__main__":
    # Test dataset loading
    from torchvision import transforms

    print("Testing DepthDataset...")

    rgb_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
    depth_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/depth_dataset"

    transform = transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DepthDataset(rgb_root, depth_root, transform=transform)

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a sample
    print("\nTesting sample loading...")
    img, depth = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")

    # Test dataloader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=depth_collate_fn)

    for batch_images, batch_depths in loader:
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch depths shape: {batch_depths.shape}")
        break

    print("\n✓ All tests passed!")
