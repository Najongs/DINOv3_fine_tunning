import os
import glob
import random
import wandb
import time
import argparse
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm

from dataset import (
    RobotPoseSequenceDataset,
    KeypointOcclusionAugmentor,
    robot_sequence_collate_fn,
    _scale_points,
    IMAGE_RESOLUTION,
    HEATMAP_SIZE
)
from model import DINOv3MotionPoseEstimator
from kinematics import get_robot_kinematics
from utils import (
    setup_ddp,
    cleanup_ddp,
    get_max_preds,
    solve_pnp_from_fk,
    transform_robot_to_camera,
    compute_masked_loss,
    save_checkpoints,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

HEATMAP_CONF_THRESHOLD = 0.5
SEQUENCE_LENGTH = 2

# ======================= 메인 학습 함수 =======================
def main(args): 
    rank, local_rank, world_size = setup_ddp()
    save_thread = None
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 16  # Per-GPU batch size, might need to be smaller due to sequence length
    EPOCHS = 100
    VAL_RATIO = 0.1
    NUM_WORKERS = 4  # Per-GPU workers

    ablation_mode = args.ablation_mode
    WANDB_PROJECT = f"DINOv3_Motion_{ablation_mode}"
    CHECKPOINT_DIR = f"checkpoints_motion_{ablation_mode}"
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

    model = DINOv3MotionPoseEstimator(dino_model_name=MODEL_NAME, heatmap_size=HEATMAP_SIZE, ablation_mode=ablation_mode)
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=False,
                static_graph=True)

    loss_fn_h  = nn.MSELoss(reduction='none')
    loss_fn_a  = nn.SmoothL1Loss(reduction='none')
    loss_fn_3D = nn.SmoothL1Loss(reduction='none')

    effective_lr = LEARNING_RATE * (world_size ** 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
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

    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if rank == 0:
        print("Loading dataset files...")
    json_files = glob.glob("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/**/*.json", recursive=True)
    if rank == 0:
        print(f"Found {len(json_files)} files.")

    if len(json_files) < SEQUENCE_LENGTH * 2:
        raise RuntimeError(f"Dataset must contain at least {SEQUENCE_LENGTH*2} samples for sequences.")

    # For sequence dataset, files MUST be sorted. Shuffling is handled by the sampler.
    json_files = sorted(json_files)
    
    # Splitting must happen before creating the dataset instance
    split_idx = int(len(json_files) * (1 - VAL_RATIO))
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    occlusion_augmentor = KeypointOcclusionAugmentor(
        prob=0.45,
        min_occlusions=1,
        max_occlusions=3,
        min_patch_ratio=0.06,
        max_patch_ratio=0.2,
        occluded_confidence=0.15,
    )

    train_dataset = RobotPoseSequenceDataset(train_files, transform, sequence_length=SEQUENCE_LENGTH, occlusion_augmentor=occlusion_augmentor)
    val_dataset = RobotPoseSequenceDataset(val_files, transform, sequence_length=SEQUENCE_LENGTH)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, collate_fn=robot_sequence_collate_fn, sampler=train_sampler,
                              persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, collate_fn=robot_sequence_collate_fn, sampler=val_sampler,
                            persistent_workers=True if NUM_WORKERS > 0 else False)
    
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(project=WANDB_PROJECT, name=f"run_motion_{ablation_mode}", config={
            "base_learning_rate": LEARNING_RATE,
            "effective_learning_rate": effective_lr,
            "per_gpu_batch_size": BATCH_SIZE,
            "total_batch_size": BATCH_SIZE * world_size,
            "epochs": EPOCHS,
            "world_size": world_size,
            "num_workers": NUM_WORKERS,
            "sequence_length": SEQUENCE_LENGTH,
            "ablation_mode": ablation_mode
        }, resume="allow")

    weight_h, weight_a, weight_3d = 3.0, 1.0, 2.0

    for epoch in range(start_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for batch in pbar:
            if batch is None: # handle invalid batches from collate_fn
                continue
            (image_sequences, gt_heatmaps, gt_angles, gt_class, gt_3d_points_padded, K, dist, 
             joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_confidences) = batch

            image_sequences = image_sequences.to(local_rank)
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
                pred_heatmaps, pred_angles = model(image_sequences)

            pred_heatmaps_np = pred_heatmaps.detach().cpu().numpy()
            pred_kpts_heatmap, pred_heatmap_conf = get_max_preds(pred_heatmaps_np)
            pred_heatmap_conf = torch.from_numpy(pred_heatmap_conf.squeeze(-1)).to(local_rank)
            pred_visibility_mask = (pred_heatmap_conf >= HEATMAP_CONF_THRESHOLD).float()
            min_joints = min(joint_confidences.shape[1], pred_visibility_mask.shape[1])
            joint_confidences[:, :min_joints] = joint_confidences[:, :min_joints] * pred_visibility_mask[:, :min_joints]
            pred_visibility_np = pred_visibility_mask.detach().cpu().numpy()

            pred_3d_list = []
            for s in range(image_sequences.size(0)):
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
                        visibility_mask=pred_visibility_np[s]
                    )
                    if ok:
                        joint_coords_cam = transform_robot_to_camera(joint_coords_robot, rvec, tvec)
                    else:
                        joint_coords_cam = np.zeros_like(joint_coords_robot)
                except Exception:
                    joint_coords_cam = np.zeros_like(joint_coords_robot)

                pred_3d_list.append(torch.tensor(joint_coords_cam, device=image_sequences.device))

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
                total_loss, loss_dict = compute_masked_loss(
                    pred_heatmaps, gt_heatmaps,
                    pred_angles, gt_angles,
                    pred_3d, gt_3d,
                    joint_lengths, angle_lengths, point_lengths,
                    joint_confidences,
                    loss_fn_h, loss_fn_a, loss_fn_3D,
                    weight_h, weight_a, weight_3d
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_tensor = total_loss.detach().clone()
            torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
            train_loss += loss_tensor.item() / world_size

            if rank == 0:
                pbar.set_postfix(loss=total_loss.item() / world_size)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for batch in val_pbar:
                if batch is None:
                    continue
                (image_sequences, gt_heatmaps, gt_angles, gt_class, gt_3d_points_padded, K, dist, 
                 joint_lengths, angle_lengths, point_lengths, orig_img_sizes, joint_confidences) = batch

                image_sequences = image_sequences.to(local_rank)
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
                    pred_heatmaps, pred_angles = model(image_sequences)

                pred_heatmaps_np = pred_heatmaps.detach().cpu().numpy()
                pred_kpts_heatmap, pred_heatmap_conf = get_max_preds(pred_heatmaps_np)
                pred_heatmap_conf = torch.from_numpy(pred_heatmap_conf.squeeze(-1)).to(local_rank)
                pred_visibility_mask = (pred_heatmap_conf >= HEATMAP_CONF_THRESHOLD).float()
                min_joints = min(joint_confidences.shape[1], pred_visibility_mask.shape[1])
                joint_confidences[:, :min_joints] = joint_confidences[:, :min_joints] * pred_visibility_mask[:, :min_joints]
                pred_visibility_np = pred_visibility_mask.detach().cpu().numpy()

                pred_3d_list = []
                for s in range(image_sequences.size(0)):
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
                            visibility_mask=pred_visibility_np[s]
                        )
                        if ok:
                            joint_coords_cam = transform_robot_to_camera(joint_coords_robot, rvec, tvec)
                        else:
                            joint_coords_cam = np.zeros_like(joint_coords_robot)
                    except Exception:
                        joint_coords_cam = np.zeros_like(joint_coords_robot)

                    pred_3d_list.append(torch.tensor(joint_coords_cam, device=image_sequences.device))

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
                    total_loss, loss_dict = compute_masked_loss(
                        pred_heatmaps, gt_heatmaps,
                        pred_angles, gt_angles,
                        pred_3d, gt_3d,
                        joint_lengths, angle_lengths, point_lengths,
                        joint_confidences,
                        loss_fn_h, loss_fn_a, loss_fn_3D,
                        weight_h, weight_a, weight_3d
                    )

                loss_tensor = total_loss.detach().clone()
                torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size

                if rank == 0:
                    val_pbar.set_postfix(loss=loss_tensor.item() / world_size)

        
        scheduler.step()
        
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
    parser = argparse.ArgumentParser(description="DINOv3 Motion Pose Estimation Training")
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='combined',
        choices=['combined', 'dino_only', 'dino_conv_only', 'combined_conv', 'siglip_only', 'siglip_combined', 'siglip2_only', 'siglip2_combined'],
        help="Select the ablation mode: 'combined' (ViT), 'dino_only', 'dino_conv_only', 'combined_conv', 'siglip_only', 'siglip_combined', 'siglip2_only', or 'siglip2_combined'"
    )
    args = parser.parse_args()

    main(args)