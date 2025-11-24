import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

FEATURE_DIM = 512
NUM_ANGLES = 9
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

class DINOv3Backbone(nn.Module):
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
        else: # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=4, nhead=8, num_decoder_layers=2):
        super().__init__()
        
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=input_dim * 4, # 일반적인 설정
            dropout=0.1, 
            activation='gelu',
            batch_first=True  # (batch, seq, feature) 입력을 위함
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.angle_predictor = nn.Sequential(
            nn.LayerNorm(input_dim * num_queries),
            nn.Linear(input_dim * num_queries, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_angles)
        )
    
    def forward(self, fused_features):
        b = fused_features.size(0)
        queries = self.pose_queries.repeat(b, 1, 1)
        attn_output = self.transformer_decoder(tgt=queries, memory=fused_features)
        output_flat = attn_output.flatten(start_dim=1)
        return self.angle_predictor(output_flat)

class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bias=True를 추가하여 PyTorch CUDA kernel의 gradient stride 문제 완화
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
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/2
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/4
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/8
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
    def forward(self, x):
        feat_4 = self.conv_block1(x)  # 1/4 스케일 특징
        feat_8 = self.conv_block2(feat_4) # 1/8 스케일 특징
        return feat_4, feat_8 # 다른 해상도의 특징들을 반환

class FusedUpsampleBlock(nn.Module):
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
                size=x.shape[-2:], # target H, W
                mode='bilinear', 
                align_corners=False
            )

        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)

class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.heatmap_predictor = nn.Conv2d(64, num_joints, kernel_size=3, padding=1)

    def forward(self, dino_features, cnn_features):
        cnn_feat_4, cnn_feat_8 = cnn_features
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        
        x = self.token_fuser(x)
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        x = self.final_upsample(x)
        heatmaps = self.heatmap_predictor(x)
        
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        
class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, ablation_mode=None):
        super().__init__()
        # self.ablation_mode = ablation_mode
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name)
        
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size
        
        self.cnn_stem = LightCNNStem()
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, heatmap_size=heatmap_size)
        self.angle_head = JointAngleHead(input_dim=feature_dim)      # 기존 Attention 헤드
        
    def forward(self, image_tensor_batch):
        dino_features = self.backbone(image_tensor_batch) # 항상 3D 텐서
        cnn_stem_features = self.cnn_stem(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features)
        
        return predicted_heatmaps, predicted_angles

class DINOv3MotionPoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, ablation_mode=None):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name)
        
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        self.temporal_processor = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.cnn_stem = LightCNNStem()
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, heatmap_size=heatmap_size)
        self.angle_head = JointAngleHead(input_dim=feature_dim)
        
    def forward(self, image_sequence_batch):
        b, t, c, h, w = image_sequence_batch.shape
        
        # 1. Reshape for backbone processing
        reshaped_images = image_sequence_batch.reshape(b * t, c, h, w)
        
        # 2. Extract DINO features for all images in the sequence
        dino_features = self.backbone(reshaped_images) # (B*T, N, D)
        
        # 3. Process features temporally with LSTM
        _bt, n, d = dino_features.shape
        # We process each patch token's sequence over time
        # Reshape so that time is the sequence length for the LSTM
        dino_features_temporal = dino_features.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        
        lstm_out, _ = self.temporal_processor(dino_features_temporal) # (B*N, T, D)
        
        # Get the output of the last time step
        last_step_features = lstm_out[:, -1, :] # (B*N, D)
        
        # Reshape back to (B, N, D) for the prediction heads
        temporal_context_features = last_step_features.reshape(b, n, d)

        # 4. Extract CNN features for the *last* frame only (for skip connections)
        last_frame_image = image_sequence_batch[:, -1, :, :, :]
        cnn_stem_features = self.cnn_stem(last_frame_image)
        
        # 5. Predict using the temporally-aware features
        predicted_heatmaps = self.keypoint_head(temporal_context_features, cnn_stem_features)
        predicted_angles = self.angle_head(temporal_context_features)
        
        return predicted_heatmaps, predicted_angles


class DINOv3MotionVectorEstimator(nn.Module):
    """
    Estimator that predicts both the current pose (heatmaps, angles) and 
    the motion vector (change in angles) for the next time step.
    """
    def __init__(self, dino_model_name, heatmap_size, ablation_mode=None):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name)
        
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        self.temporal_processor = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.cnn_stem = LightCNNStem()
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, heatmap_size=heatmap_size)
        self.angle_head = JointAngleHead(input_dim=feature_dim)
        
        # New head for predicting the motion vector (angle deltas)
        self.angle_delta_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, NUM_ANGLES)
        )
        
    def forward(self, image_sequence_batch):
        b, t, c, h, w = image_sequence_batch.shape
        
        reshaped_images = image_sequence_batch.reshape(b * t, c, h, w)
        dino_features = self.backbone(reshaped_images)
        
        _bt, n, d = dino_features.shape
        dino_features_temporal = dino_features.view(b, t, n, d).permute(0, 2, 1, 3).reshape(b * n, t, d)
        
        lstm_out, _ = self.temporal_processor(dino_features_temporal)
        
        last_step_features = lstm_out[:, -1, :]
        
        temporal_context_features = last_step_features.reshape(b, n, d)

        last_frame_image = image_sequence_batch[:, -1, :, :, :]
        cnn_stem_features = self.cnn_stem(last_frame_image)
        
        # Predict current pose (t)
        predicted_heatmaps = self.keypoint_head(temporal_context_features, cnn_stem_features)
        predicted_angles = self.angle_head(temporal_context_features)
        
        # Predict motion vector (t -> t+1)
        # We use the globally pooled features from the temporal context for delta prediction
        pooled_temporal_features = temporal_context_features.mean(dim=1) # (B, D)
        predicted_angle_deltas = self.angle_delta_head(pooled_temporal_features)
        
        return predicted_heatmaps, predicted_angles, predicted_angle_deltas
