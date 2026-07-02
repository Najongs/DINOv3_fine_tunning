import os
import glob
import random
import wandb
import time
import argparse
import threading
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

from dataset import (
    RobotPoseDataset,
    KeypointOcclusionAugmentor,
    robot_collate_fn,
    _scale_points,
    IMAGE_RESOLUTION,
    HEATMAP_SIZE
)
from model import DINOv3PoseEstimator
from kinematics import get_robot_kinematics
from utils import (
    setup_ddp,
    cleanup_ddp,
    get_max_preds,
    solve_pnp_from_fk,
    transform_robot_to_camera,
    save_checkpoints,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ======================= 간소화된 Loss 함수 =======================
def compute_simple_loss(pred_heatmaps, gt_heatmaps, pred_angles, gt_angles,
                       pred_3d, gt_3d, joint_lengths, angle_lengths, point_lengths,
                       loss_fn_h, loss_fn_a, loss_fn_3D, weight_h, weight_a, weight_3d,
                       joint_confidences=None):
    """
    Simplified loss with length-based masking.
    If joint_confidences is provided (e.g., occlusion), it down-weights those joints/points.
    """
    B = pred_heatmaps.size(0)

    # 1. Heatmap Loss (length masking + optional occlusion mask)
    loss_h_per_sample = []
    for i in range(B):
        n_joints = joint_lengths[i].item()
        pred_h = pred_heatmaps[i, :n_joints]
        gt_h = gt_heatmaps[i, :n_joints]
        if joint_confidences is not None:
            conf = joint_confidences[i, :n_joints].view(-1, 1, 1)
            loss_h_per_sample.append((loss_fn_h(pred_h, gt_h) * conf).sum() / conf.sum().clamp_min(1.0))
        else:
            loss_h_per_sample.append(loss_fn_h(pred_h, gt_h).mean())
    loss_h = torch.stack(loss_h_per_sample).mean()

    # 2. Angle Loss (length masking + optional occlusion mask transfer)
    loss_a_per_sample = []
    for i in range(B):
        n_angles = angle_lengths[i].item()
        pred_a = pred_angles[i, :n_angles]
        gt_a = gt_angles[i, :n_angles]
        if joint_confidences is not None:
            max_transfer = min(n_angles, joint_confidences.shape[1])
            conf = joint_confidences[i, :max_transfer]
            loss_a_per_sample.append((loss_fn_a(pred_a, gt_a)[:max_transfer] * conf).sum() / conf.sum().clamp_min(1.0))
        else:
            loss_a_per_sample.append(loss_fn_a(pred_a, gt_a).mean())
    loss_a = torch.stack(loss_a_per_sample).mean()

    # 3. 3D Loss (length masking + optional occlusion mask)
    loss_3d_per_sample = []
    for i in range(B):
        n_points = point_lengths[i].item()
        pred_3d_i = pred_3d[i, :n_points]
        gt_3d_i = gt_3d[i, :n_points]
        if joint_confidences is not None:
            conf = joint_confidences[i, :n_points].view(-1, 1)
            loss_3d_per_sample.append((loss_fn_3D(pred_3d_i, gt_3d_i) * conf).sum() / conf.sum().clamp_min(1.0))
        else:
            loss_3d_per_sample.append(loss_fn_3D(pred_3d_i, gt_3d_i).mean())
    loss_3d = torch.stack(loss_3d_per_sample).mean()

    # Total weighted loss
    total_loss = weight_h * loss_h + weight_a * loss_a + weight_3d * loss_3d

    loss_dict = {
        'loss_h': loss_h,
        'loss_a': loss_a,
        'loss_3d': loss_3d
    }

    return total_loss, loss_dict

# ======================= Heatmap 시각화 함수 =======================
def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    정규화된 이미지 텐서를 원본 범위로 변환
    """
    img = img_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)
    return img

def visualize_heatmaps_before_training(model, data_loader, device, save_dir, rank=0, num_samples=4):
    """
    학습 시작 전에 원본 이미지, GT heatmap, predicted heatmap을 시각화합니다.
    """
    if rank != 0:
        return  # Only rank 0 visualizes

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Get a batch from the data loader
    batch_iter = iter(data_loader)
    batch = next(batch_iter)

    (images, gt_heatmaps, gt_angles, gt_class, gt_3d_points_padded, K, dist,
     joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_confidences) = batch

    images = images.to(device)
    gt_heatmaps = gt_heatmaps.to(device)

    # Get predictions
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_heatmaps, pred_angles = model(images)

    # Visualize only the first few samples
    num_samples = min(num_samples, images.size(0))

    for sample_idx in range(num_samples):
        n_joints = joint_lengths[sample_idx].item()

        # Select a few representative joints to visualize (max 6)
        joints_to_vis = min(6, n_joints)

        # Create figure with 3 rows: original image, GT heatmaps, predicted heatmaps
        fig, axes = plt.subplots(3, joints_to_vis, figsize=(joints_to_vis * 3, 9))
        if joints_to_vis == 1:
            axes = axes.reshape(3, 1)

        # Denormalize original image
        orig_img = denormalize_image(images[sample_idx].cpu())
        orig_img_np = orig_img.permute(1, 2, 0).numpy()

        for joint_idx in range(joints_to_vis):
            # Row 0: Original image (same for all columns)
            axes[0, joint_idx].imshow(orig_img_np)
            axes[0, joint_idx].set_title(f'Original Image (Joint {joint_idx})')
            axes[0, joint_idx].axis('off')

            # Row 1: GT heatmap
            gt_hm = gt_heatmaps[sample_idx, joint_idx].cpu().numpy()
            axes[1, joint_idx].imshow(gt_hm, cmap='hot', interpolation='nearest')
            axes[1, joint_idx].set_title(f'GT Heatmap {joint_idx}')
            axes[1, joint_idx].axis('off')

            # Row 2: Predicted heatmap
            pred_hm = pred_heatmaps[sample_idx, joint_idx].detach().cpu().numpy()
            axes[2, joint_idx].imshow(pred_hm, cmap='hot', interpolation='nearest')
            axes[2, joint_idx].set_title(f'Pred Heatmap {joint_idx}')
            axes[2, joint_idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'heatmap_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved heatmap visualization: {save_path}")

    model.train()

# ======================= 메인 학습 함수 =======================
def main(args):
    rank, local_rank, world_size = setup_ddp()
    save_thread = None
    BASE_LR = 1e-4
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 5
    OCCLUSION_START_EPOCH = 225  # enable occlusion after some initial convergence
    PHOTO_AUG_START_EPOCH = 250   # enable light photometric aug after initial convergence
    PHOTO_AUG_PROB = 0.4
    BATCH_SIZE = 32  # Per-GPU batch size
    EPOCHS = 300
    VAL_RATIO = 0.05
    NUM_WORKERS = 4  # Per-GPU workers

    ablation_mode = args.ablation_mode
    WANDB_PROJECT = f"DINOv3_Simple_{ablation_mode}"
    CHECKPOINT_DIR = f"checkpoints_simple_{ablation_mode}"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

    # 모델 이름 결정
    if 'vit' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    elif 'conv' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'
    elif 'siglip2' in ablation_mode:
        MODEL_NAME = 'google/siglip2-base-patch16-224'
    elif 'siglip' in ablation_mode:
        MODEL_NAME = 'google/siglip-base-patch16-224'
    else:
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

    start_epoch = 0
    best_val_loss = float('inf')

    model = DINOv3PoseEstimator(dino_model_name=MODEL_NAME, heatmap_size=HEATMAP_SIZE, ablation_mode=ablation_mode)
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=False,
                static_graph=True)

    loss_fn_h  = nn.MSELoss(reduction='none')
    loss_fn_a  = nn.SmoothL1Loss(reduction='none')
    loss_fn_3D = nn.SmoothL1Loss(reduction='none')

    effective_lr = BASE_LR * (world_size ** 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)

    def lr_lambda(current_epoch):
        warmup_factor = MIN_LR / effective_lr
        if current_epoch < WARMUP_EPOCHS:
            return warmup_factor + (1 - warmup_factor) * (current_epoch / max(1, WARMUP_EPOCHS))
        progress = (current_epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return (MIN_LR / effective_lr) + (1 - (MIN_LR / effective_lr)) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists(LATEST_CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=map_location)

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        if rank == 0:
            print(f"✅ 체크포인트({CHECKPOINT_PATH})를 성공적으로 불러왔습니다.")
            print(f"   - {start_epoch - 1} 에포크까지 학습 완료됨. {start_epoch} 에포크부터 학습을 재개합니다.")
            print(f"   - 현재까지 Best Val Loss: {best_val_loss:.6f}")

    elif os.path.exists(CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.module.load_state_dict(state_dict)
        if rank == 0:
            print(f"✅ 체크포인트에서 모델 가중치를 성공적으로 불러왔습니다: {CHECKPOINT_PATH}")

    else:
        if rank == 0:
            print(f"ℹ️ 체크포인트 파일({LATEST_CHECKPOINT_PATH})이 없으므로, 처음부터 학습을 시작합니다.")

    def make_train_transform(use_photo_aug):
        t = [transforms.Resize(IMAGE_RESOLUTION)]
        if use_photo_aug:
            t.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)],
                    p=PHOTO_AUG_PROB
                )
            )
        t.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transforms.Compose(t)

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if rank == 0:
        print("Loading dataset files...")
    json_files = glob.glob("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/**/*.json", recursive=True)
    if rank == 0:
        print(f"Found {len(json_files)} files.")

    if len(json_files) < 2:
        raise RuntimeError("Dataset must contain at least two samples to create train/val splits.")

    json_files = sorted(json_files)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * (1 - VAL_RATIO))
    split_idx = min(max(split_idx, 1), len(json_files) - 1)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    def build_train_loader(use_occlusion, use_photo_aug):
        dataset = RobotPoseDataset(
            train_files,
            make_train_transform(use_photo_aug),
            sigma=5.0,
            occlusion_augmentor=KeypointOcclusionAugmentor(
                prob=0.3,
                min_occlusions=1,
                max_occlusions=3,
                min_patch_ratio=0.1,
                max_patch_ratio=0.2,
                occluded_confidence=0.15,
            ) if use_occlusion else None
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, collate_fn=robot_collate_fn, sampler=sampler,
            persistent_workers=False if NUM_WORKERS > 0 else False
        )
        return dataset, sampler, loader

    occlusion_enabled = start_epoch >= OCCLUSION_START_EPOCH
    photo_aug_enabled = start_epoch >= PHOTO_AUG_START_EPOCH
    train_dataset, train_sampler, train_loader = build_train_loader(occlusion_enabled, photo_aug_enabled)

    val_dataset = RobotPoseDataset(val_files, val_transform, sigma=5.0)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, collate_fn=robot_collate_fn, sampler=val_sampler,
                            persistent_workers=True if NUM_WORKERS > 0 else False)

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(project=WANDB_PROJECT, name=f"run_simple_{ablation_mode}", config={
            "base_learning_rate": BASE_LR,
            "min_learning_rate": MIN_LR,
            "effective_learning_rate": effective_lr,
            "per_gpu_batch_size": BATCH_SIZE,
            "total_batch_size": BATCH_SIZE * world_size,
            "epochs": EPOCHS,
            "world_size": world_size,
            "num_workers": NUM_WORKERS,
            "ablation_mode": ablation_mode,
            "note": "Simplified training; 3D loss eval-only; occlusion + photo aug after warmup"
        }, resume="allow")

    # ======================= Heatmap 시각화 (학습 전) =======================
    if rank == 0:
        print("\n" + "="*60)
        print("🔍 Visualizing heatmaps before training...")
        print("="*60)
        vis_dir = os.path.join(CHECKPOINT_DIR, "heatmap_visualizations")
        visualize_heatmaps_before_training(
            model=model.module if hasattr(model, 'module') else model,
            data_loader=train_loader,
            device=local_rank,
            save_dir=vis_dir,
            rank=rank,
            num_samples=4
        )
        print("="*60 + "\n")

    weight_h, weight_a, weight_3d = 2.0, 1.0, 0.0  # 3D loss excluded from optimization

    for epoch in range(start_epoch, EPOCHS):
        scheduler.step()
        if (not occlusion_enabled) and epoch >= OCCLUSION_START_EPOCH:
            if rank == 0:
                print(f"⚙️ Enabling occlusion augmentation from epoch {epoch}")
            occlusion_enabled = True
            train_dataset, train_sampler, train_loader = build_train_loader(occlusion_enabled, photo_aug_enabled)
        if (not photo_aug_enabled) and epoch >= PHOTO_AUG_START_EPOCH:
            if rank == 0:
                print(f"🎨 Enabling photometric augmentation from epoch {epoch}")
            photo_aug_enabled = True
            train_dataset, train_sampler, train_loader = build_train_loader(occlusion_enabled, photo_aug_enabled)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for batch in pbar:
            (images, gt_heatmaps, gt_angles, gt_class, gt_3d_points_padded, K, dist,
             joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_confidences) = batch

            images      = images.to(local_rank)
            gt_heatmaps = gt_heatmaps.to(local_rank)
            gt_angles   = gt_angles.to(local_rank)
            gt_3d       = gt_3d_points_padded.to(local_rank)
            joint_lengths  = joint_lengths.to(local_rank)
            angle_lengths  = angle_lengths.to(local_rank)
            point_lengths  = point_lengths.to(local_rank)
            K          = K.to(local_rank)
            dist       = dist.to(local_rank)
            orig_img_sizes = orig_img_sizes.to(local_rank)
            joint_confidences = joint_confidences.to(local_rank)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred_heatmaps, pred_angles = model(images)

            # Simplified 3D prediction without visibility masking
            # For PnP, follow the previous behavior: use GT heatmap peaks instead of predictions
            gt_heatmaps_np = gt_heatmaps.detach().cpu().numpy()
            pred_kpts_heatmap, _ = get_max_preds(gt_heatmaps_np)

            pred_3d_list = []
            for s in range(images.size(0)):
                robot = get_robot_kinematics(gt_class[s])
                joint_angles = robot._truncate_angles(pred_angles[s].detach().cpu().numpy())
                joint_coords_robot = robot.forward_kinematics(joint_angles)

                heatmap_size = (pred_heatmaps.shape[3], pred_heatmaps.shape[2])
                img_size = orig_img_sizes[s].cpu().numpy()
                img_kpts2d = _scale_points(pred_kpts_heatmap[s], from_size=heatmap_size, to_size=img_size)

                K_s = K[s].detach().cpu().numpy()
                dist_s = dist[s].detach().cpu().numpy()

                try:
                    # Use all points without visibility mask
                    rvec, tvec, ok = solve_pnp_from_fk(
                        joint_coords_robot, img_kpts2d, K_s, dist_s,
                        visibility_mask=None
                    )
                    if ok:
                        joint_coords_cam = transform_robot_to_camera(joint_coords_robot, rvec, tvec)
                    else:
                        joint_coords_cam = np.zeros_like(joint_coords_robot)
                except Exception:
                    joint_coords_cam = np.zeros_like(joint_coords_robot)

                pred_3d_list.append(torch.tensor(joint_coords_cam, device=images.device))

            padded_pred_3d_list = []
            max_len = gt_3d.shape[1]
            for p in pred_3d_list:
                pad_len = max_len - p.shape[0]
                if pad_len > 0:
                    padded_p = F.pad(p, (0, 0, 0, pad_len), "constant", 0)
                    padded_pred_3d_list.append(padded_p)
                else:
                    padded_pred_3d_list.append(p[:max_len])

            pred_3d = torch.stack(padded_pred_3d_list, dim=0).to(local_rank)

            # Use simplified loss function
            with torch.amp.autocast('cuda'):
                total_loss, loss_dict = compute_simple_loss(
                    pred_heatmaps, gt_heatmaps,
                    pred_angles, gt_angles,
                    pred_3d, gt_3d,
                    joint_lengths, angle_lengths, point_lengths,
                    loss_fn_h, loss_fn_a, loss_fn_3D,
                    weight_h, weight_a, weight_3d,
                    joint_confidences
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_tensor = total_loss.detach().clone()
            torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
            train_loss += loss_tensor.item() / world_size

            if rank == 0:
                pbar.set_postfix(loss=total_loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for batch in val_pbar:
                (images, gt_heatmaps, gt_angles, gt_class, gt_3d_points_padded, K, dist,
                 joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_confidences) = batch

                images      = images.to(local_rank)
                gt_heatmaps = gt_heatmaps.to(local_rank)
                gt_angles   = gt_angles.to(local_rank)
                gt_3d       = gt_3d_points_padded.to(local_rank)
                joint_lengths  = joint_lengths.to(local_rank)
                angle_lengths  = angle_lengths.to(local_rank)
                point_lengths  = point_lengths.to(local_rank)
                K          = K.to(local_rank)
                dist       = dist.to(local_rank)
                orig_img_sizes = orig_img_sizes.to(local_rank)
                joint_confidences = joint_confidences.to(local_rank)

                with torch.amp.autocast('cuda'):
                    pred_heatmaps, pred_angles = model(images)

                # Use GT heatmap peaks for PnP to mirror previous approach
                gt_heatmaps_np = gt_heatmaps.detach().cpu().numpy()
                pred_kpts_heatmap, _ = get_max_preds(gt_heatmaps_np)

                pred_3d_list = []
                for s in range(images.size(0)):
                    robot = get_robot_kinematics(gt_class[s])
                    joint_angles = robot._truncate_angles(pred_angles[s].detach().cpu().numpy())
                    joint_coords_robot = robot.forward_kinematics(joint_angles)

                    heatmap_size = (pred_heatmaps.shape[3], pred_heatmaps.shape[2])
                    img_size = orig_img_sizes[s].cpu().numpy()
                    img_kpts2d = _scale_points(pred_kpts_heatmap[s], from_size=heatmap_size, to_size=img_size)

                    K_s = K[s].detach().cpu().numpy()
                    dist_s = dist[s].detach().cpu().numpy()

                    try:
                        rvec, tvec, ok = solve_pnp_from_fk(
                            joint_coords_robot, img_kpts2d, K_s, dist_s,
                            visibility_mask=None
                        )
                        if ok:
                            joint_coords_cam = transform_robot_to_camera(joint_coords_robot, rvec, tvec)
                        else:
                            joint_coords_cam = np.zeros_like(joint_coords_robot)
                    except Exception:
                        joint_coords_cam = np.zeros_like(joint_coords_robot)

                    pred_3d_list.append(torch.tensor(joint_coords_cam, device=images.device))

                padded_pred_3d_list = []
                max_len = gt_3d.shape[1]
                for p in pred_3d_list:
                    pad_len = max_len - p.shape[0]
                    if pad_len > 0:
                        padded_p = F.pad(p, (0, 0, 0, pad_len), "constant", 0)
                        padded_pred_3d_list.append(padded_p)
                    else:
                        padded_pred_3d_list.append(p[:max_len])

                pred_3d = torch.stack(padded_pred_3d_list, dim=0).to(local_rank)

                with torch.amp.autocast('cuda'):
                    total_loss, loss_dict = compute_simple_loss(
                        pred_heatmaps, gt_heatmaps,
                        pred_angles, gt_angles,
                        pred_3d, gt_3d,
                        joint_lengths, angle_lengths, point_lengths,
                        loss_fn_h, loss_fn_a, loss_fn_3D,
                        weight_h, weight_a, weight_3d,
                        joint_confidences
                    )

                loss_tensor = total_loss.detach().clone()
                torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size

                if rank == 0:
                    val_pbar.set_postfix(loss=loss_tensor.item())


        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "loss_h": loss_dict['loss_h'].item(),
                "loss_a": loss_dict['loss_a'].item(),
                "loss_3d": loss_dict['loss_3d'].item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"✨ New best model saved for '{ablation_mode}' with val_loss: {best_val_loss:.6f}")

            if save_thread is not None and save_thread.is_alive():
                print(f"Epoch {epoch+1}: Previous save is still running. Skipping this save.")
            else:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                }

                save_thread = threading.Thread(
                    target=save_checkpoints,
                    args=(checkpoint_data, model.module.state_dict(), CHECKPOINT_DIR, is_best)
                )
                save_thread.start()

    if rank == 0:
        if save_thread is not None and save_thread.is_alive():
            print("Waiting for final checkpoint save to complete...")
            save_thread.join()
        wandb.finish()
    cleanup_ddp()

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description="DINOv3 Pose Estimation - Simplified Training")
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='dino_only',
        choices=['combined', 'dino_only', 'dino_conv_only', 'combined_conv', 'siglip_only', 'siglip_combined', 'siglip2_only', 'siglip2_combined'],
        help="Select the ablation mode"
    )
    args = parser.parse_args()

    main(args)
