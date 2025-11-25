import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

# Human Pose Configuration
HUMAN_NUM_KEYPOINTS = 17  # COCO format: 17 keypoints for human pose

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


class HumanPoseKeypointHead(nn.Module):
    """
    Human pose estimation head using UNet-style architecture with ViT features.
    Predicts 2D heatmaps for human keypoints (COCO format: 17 keypoints).
    """
    def __init__(self, input_dim=768, num_keypoints=HUMAN_NUM_KEYPOINTS, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints

        # Token fusion for DINOv3 features
        self.token_fuser = TokenFuser(input_dim, 256)

        # UNet-style decoder
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)

        # Final upsampling and heatmap prediction
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.heatmap_predictor = nn.Conv2d(64, num_keypoints, kernel_size=3, padding=1)

    def forward(self, dino_features, cnn_features):
        """
        Args:
            dino_features: (B, N, D) - DINOv3 patch tokens
            cnn_features: tuple of (feat_4, feat_8) - CNN stem features

        Returns:
            heatmaps: (B, num_keypoints, H, W) - predicted keypoint heatmaps
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
        x = self.token_fuser(x)
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        x = self.final_upsample(x)
        heatmaps = self.heatmap_predictor(x)

        # Resize to target heatmap size
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)


class DINOv3HumanPoseEstimator(nn.Module):
    """
    Complete human pose estimation model using frozen DINOv3 backbone.
    Predicts 2D heatmaps for human keypoints.
    """
    def __init__(self, dino_model_name, heatmap_size=(512, 512), num_keypoints=HUMAN_NUM_KEYPOINTS):
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

        # Human pose keypoint head
        self.keypoint_head = HumanPoseKeypointHead(
            input_dim=feature_dim,
            num_keypoints=num_keypoints,
            heatmap_size=heatmap_size
        )

    def forward(self, image_tensor_batch):
        """
        Args:
            image_tensor_batch: (B, 3, H, W) - input images

        Returns:
            predicted_heatmaps: (B, num_keypoints, H_heatmap, W_heatmap) - predicted keypoint heatmaps
        """
        # Extract features from frozen backbone
        dino_features = self.backbone(image_tensor_batch)  # (B, N, D)

        # Extract multi-scale CNN features
        cnn_stem_features = self.cnn_stem(image_tensor_batch)  # (feat_4, feat_8)

        # Predict keypoint heatmaps
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)

        return predicted_heatmaps


if __name__ == "__main__":
    # Test the model
    print("Testing DINOv3HumanPoseEstimator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/dinov2-base"  # or "facebook/dinov2-large"
    heatmap_size = (512, 512)

    model = DINOv3HumanPoseEstimator(
        dino_model_name=model_name,
        heatmap_size=heatmap_size,
        num_keypoints=17
    ).to(device)

    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output heatmap shape: {output.shape}")
    print(f"Expected: ({batch_size}, 17, {heatmap_size[0]}, {heatmap_size[1]})")
    print("Model test passed!")
