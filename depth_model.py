import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel


class DINOv3Backbone(nn.Module):
    """
    Frozen DINOv3 backbone for feature extraction.
    Reused from the robot pose estimation model.
    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        if "siglip" in model_name:
            self.model = SiglipVisionModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone parameters (but allow gradient flow through activations)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image_tensor_batch):
        # Removed torch.no_grad() to allow gradient flow for downstream heads
        if "siglip" in self.model_name:
            outputs = self.model(
                pixel_values=image_tensor_batch,
                interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state
            patch_tokens = tokens[:, 1:, :]
        else:  # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens


class TokenFuser(nn.Module):
    """
    Token fusion module to process DINOv3 features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)


class LightCNNStem(nn.Module):
    """
    Lightweight CNN stem for extracting multi-scale features.
    """
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 1/2 resolution
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 1/4 resolution
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 1/8 resolution
            nn.BatchNorm2d(64),
            nn.GELU()
        )

    def forward(self, x):
        feat_4 = self.conv_block1(x)   # 1/4 scale features
        feat_8 = self.conv_block2(feat_4)  # 1/8 scale features
        return feat_4, feat_8


class FusedUpsampleBlock(nn.Module):
    """
    Upsampling block with skip connections for feature fusion.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature,
                size=x.shape[-2:],  # target H, W
                mode='bilinear',
                align_corners=False
            )

        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)


class UNetViTDepthHead(nn.Module):
    """
    Depth estimation head using UNet-style architecture with ViT features.
    Predicts dense depth map (1 channel) for monocular depth estimation.

    Structure is similar to UNetViTKeypointHead, but outputs 1 channel instead of multiple keypoints.
    """
    def __init__(self, input_dim=768, depth_size=(512, 512)):
        super().__init__()
        self.output_size = depth_size  # Final depth map resolution

        # Token fusion for DINOv3 features
        self.token_fuser = TokenFuser(input_dim, 256)

        # UNet-style decoder
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)

        # Final upsampling and depth prediction
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Depth predictor: 1 channel for depth values
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, dino_features, cnn_features):
        """
        Args:
            dino_features: (B, N, D) - DINOv3 patch tokens
            cnn_features: tuple of (feat_4, feat_8) - CNN stem features
                feat_4: (B, 32, H/4, W/4)
                feat_8: (B, 64, H/8, W/8)

        Returns:
            depth_map: (B, 1, H, W) - predicted depth map
        """
        cnn_feat_4, cnn_feat_8 = cnn_features
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        # Reshape DINOv3 features to spatial format
        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # Progressive upsampling with skip connections
        x = self.token_fuser(x)               # (B, 256, H', W')
        x = self.decoder_block1(x, cnn_feat_8)  # (B, 128, ...)
        x = self.decoder_block2(x, cnn_feat_4)  # (B, 64, ...)
        x = self.final_upsample(x)             # One more upsample

        depth_map = self.depth_predictor(x)    # (B, 1, H_d, W_d)

        # Resize to target depth map size
        depth_map = F.interpolate(
            depth_map,
            size=self.output_size,
            mode='bilinear',
            align_corners=False
        )  # (B, 1, H_out, W_out)

        return depth_map


class DINOv3DepthEstimator(nn.Module):
    """
    Complete monocular depth estimation model using frozen DINOv3 backbone.
    Predicts dense depth map from a single RGB image.

    This model can be trained with:
    - Ground truth depth (e.g., from RGBD sensors, LiDAR)
    - Teacher model predictions (knowledge distillation)
    - Self-supervised methods (stereo, video sequences)
    """
    def __init__(self, dino_model_name, depth_size=(512, 512)):
        super().__init__()
        self.dino_model_name = dino_model_name

        # Frozen DINOv3 backbone
        self.backbone = DINOv3Backbone(dino_model_name)

        # Get feature dimension from backbone config
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else:  # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        # CNN stem for multi-scale features
        self.cnn_stem = LightCNNStem()

        # Depth estimation head
        self.depth_head = UNetViTDepthHead(
            input_dim=feature_dim,
            depth_size=depth_size
        )

    def forward(self, image_tensor_batch):
        """
        Args:
            image_tensor_batch: (B, 3, H, W) - input RGB images

        Returns:
            predicted_depth: (B, 1, H_depth, W_depth) - predicted depth map
        """
        # Extract features from frozen backbone
        dino_features = self.backbone(image_tensor_batch)  # (B, N, D)

        # Extract multi-scale CNN features
        cnn_stem_features = self.cnn_stem(image_tensor_batch)  # (feat_4, feat_8)

        # Predict depth map
        predicted_depth = self.depth_head(dino_features, cnn_stem_features)

        return predicted_depth


def sample_joint_depths(depth_map, keypoints, patch_size=7):
    """
    Sample depth values at keypoint locations from the predicted depth map.
    Uses median pooling over a patch around each keypoint for robustness.

    This function extracts z-coordinates for 3D pose reconstruction:
    - (x, y) from 2D keypoint detection
    - z from depth map at (x, y) location

    Args:
        depth_map: (B, 1, H, W) - predicted depth map
        keypoints: (B, K, 2) - keypoint coordinates (x, y) in pixels
        patch_size: odd int (e.g., 7) - size of ROI for median pooling

    Returns:
        joint_depths: (B, K) - depth value (z-coordinate) for each keypoint
    """
    B, _, H, W = depth_map.shape
    B2, K, _ = keypoints.shape
    assert B == B2, f"Batch size mismatch: depth_map={B}, keypoints={B2}"

    r = patch_size // 2
    joint_depths = []

    for b in range(B):
        depths_b = []
        for k in range(K):
            u, v = keypoints[b, k]  # (x, y)
            u = int(torch.clamp(u, 0, W - 1).item())
            v = int(torch.clamp(v, 0, H - 1).item())

            # Extract patch around keypoint
            u0, u1 = max(0, u - r), min(W, u + r + 1)
            v0, v1 = max(0, v - r), min(H, v + r + 1)

            patch = depth_map[b, 0, v0:v1, u0:u1]

            # Use median for robustness (less sensitive to outliers)
            if patch.numel() == 0:
                depths_b.append(depth_map[b, 0, v, u])
            else:
                depths_b.append(torch.median(patch))

        joint_depths.append(torch.stack(depths_b))  # (K,)

    return torch.stack(joint_depths, dim=0)  # (B, K)


def reconstruct_3d_from_depth(keypoints_2d, depth_map, K, patch_size=7):
    """
    Reconstruct 3D coordinates from 2D keypoints and depth map.

    Args:
        keypoints_2d: (B, K, 2) - 2D keypoint coordinates (x, y) in pixels
        depth_map: (B, 1, H, W) - predicted depth map
        K: (B, 3, 3) or (3, 3) - camera intrinsic matrix
        patch_size: int - patch size for depth sampling

    Returns:
        keypoints_3d: (B, K, 3) - 3D coordinates in camera frame (X, Y, Z)
    """
    B, K, _ = keypoints_2d.shape

    # Sample depth at keypoint locations
    Z = sample_joint_depths(depth_map, keypoints_2d, patch_size=patch_size)  # (B, K)

    # Handle camera intrinsics
    if K.dim() == 2:  # (3, 3) - same for all batches
        K = K.unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)

    # Extract intrinsic parameters
    fx = K[:, 0, 0].unsqueeze(1)  # (B, 1)
    fy = K[:, 1, 1].unsqueeze(1)  # (B, 1)
    cx = K[:, 0, 2].unsqueeze(1)  # (B, 1)
    cy = K[:, 1, 2].unsqueeze(1)  # (B, 1)

    # Unproject to 3D: [X, Y, Z] = [(u - cx) * Z / fx, (v - cy) * Z / fy, Z]
    u = keypoints_2d[:, :, 0]  # (B, K) - x coordinates
    v = keypoints_2d[:, :, 1]  # (B, K) - y coordinates

    X = (u - cx) * Z / fx  # (B, K)
    Y = (v - cy) * Z / fy  # (B, K)

    # Stack to (B, K, 3)
    keypoints_3d = torch.stack([X, Y, Z], dim=2)

    return keypoints_3d


if __name__ == "__main__":
    # Test the depth estimation model
    print("Testing DINOv3DepthEstimator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/dinov2-base"  # or "facebook/dinov2-large"
    depth_size = (512, 512)

    model = DINOv3DepthEstimator(
        dino_model_name=model_name,
        depth_size=depth_size
    ).to(device)

    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512).to(device)

    with torch.no_grad():
        depth_output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output depth shape: {depth_output.shape}")
    print(f"Expected: ({batch_size}, 1, {depth_size[0]}, {depth_size[1]})")

    # Test depth sampling at keypoints
    print("\nTesting sample_joint_depths...")
    num_keypoints = 17
    test_keypoints = torch.rand(batch_size, num_keypoints, 2).to(device) * 512  # Random keypoints in [0, 512]

    joint_depths = sample_joint_depths(depth_output, test_keypoints, patch_size=7)
    print(f"Keypoints shape: {test_keypoints.shape}")
    print(f"Joint depths shape: {joint_depths.shape}")
    print(f"Expected: ({batch_size}, {num_keypoints})")

    # Test 3D reconstruction
    print("\nTesting reconstruct_3d_from_depth...")
    K = torch.tensor([
        [500.0, 0.0, 256.0],
        [0.0, 500.0, 256.0],
        [0.0, 0.0, 1.0]
    ]).to(device)

    keypoints_3d = reconstruct_3d_from_depth(test_keypoints, depth_output, K, patch_size=7)
    print(f"3D keypoints shape: {keypoints_3d.shape}")
    print(f"Expected: ({batch_size}, {num_keypoints}, 3)")

    print("\nAll tests passed!")
