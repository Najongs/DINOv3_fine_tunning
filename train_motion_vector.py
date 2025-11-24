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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm

from dataset import (
    RobotPoseMotionVectorDataset,
    KeypointOcclusionAugmentor,
    robot_motion_vector_collate_fn,
    _scale_points,
    IMAGE_RESOLUTION,
    HEATMAP_SIZE
)
from model import DINOv3MotionVectorEstimator
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

HEATMAP_CONF_THRESHOLD = 0.5
SEQUENCE_LENGTH = 2

def main(args): 
    rank, local_rank, world_size = setup_ddp()
    save_thread = None
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 12
    EPOCHS = 100
    VAL_RATIO = 0.1
    NUM_WORKERS = 4

    ablation_mode = args.ablation_mode
    WANDB_PROJECT = f"DINOv3_Motion_Vector_{ablation_mode}"
    CHECKPOINT_DIR = f"checkpoints_motion_vector_{ablation_mode}"
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    PRETRAINED_CHECKPOINT_DIR = f"checkpoints_motion_{ablation_mode}"
    
    if 'vit' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    elif 'conv' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'
    else:
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

    start_epoch = 0
    best_val_loss = float('inf')

    model = DINOv3MotionVectorEstimator(dino_model_name=MODEL_NAME, heatmap_size=HEATMAP_SIZE, ablation_mode=ablation_mode)
    model.to(local_rank)
    
    loss_fn_h = nn.MSELoss(reduction='none')
    loss_fn_a = nn.SmoothL1Loss(reduction='none')
    loss_fn_3D = nn.SmoothL1Loss(reduction='none')

    effective_lr = LEARNING_RATE * (world_size ** 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists(LATEST_CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if rank == 0:
            print(f"✅ Motion Vector 모델 체크포인트를 불러와 학습을 재개합니다: {LATEST_CHECKPOINT_PATH}")
    else:
        pretrained_path = os.path.join(PRETRAINED_CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(pretrained_path):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(pretrained_path, map_location=map_location)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Load weights with strict=False to allow for the new motion vector head
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if rank == 0:
                print(f"✅ 기존 Pose 모델에서 전이 학습을 시작합니다: {pretrained_path}")
                if missing_keys:
                    print("   - 다음 가중치가 없어 랜덤 초기화됩니다 (정상):", [k for k in missing_keys if 'angle_delta_head' in k])
                if unexpected_keys:
                     print("   - 체크포인트에 있지만 모델에 없는 가중치:", unexpected_keys)
        else:
            if rank == 0:
                print(f"ℹ️ 체크포인트가 없어 처음부터 학습을 시작합니다.")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    json_files = sorted(glob.glob("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/**/*.json", recursive=True))
    
    split_idx = int(len(json_files) * (1 - VAL_RATIO))
    train_files, val_files = json_files[:split_idx], json_files[split_idx:]

    occlusion_augmentor = KeypointOcclusionAugmentor()
    train_dataset = RobotPoseMotionVectorDataset(train_files, transform, sequence_length=SEQUENCE_LENGTH, occlusion_augmentor=occlusion_augmentor, max_time_interval_s=5.0)
    val_dataset = RobotPoseMotionVectorDataset(val_files, transform, sequence_length=SEQUENCE_LENGTH, max_time_interval_s=5.0)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=robot_motion_vector_collate_fn, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=robot_motion_vector_collate_fn, sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(project=WANDB_PROJECT, name=f"run_vec_{ablation_mode}", resume="allow")

    weight_h, weight_a_t, weight_ad, weight_3d = 3.0, 1.0, 1.5, 2.0

    for epoch in range(start_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for batch in pbar:
            if batch is None: continue
            (image_sequences, gt_heatmaps_t, gt_angles_t, gt_angle_deltas, gt_class, 
             gt_3d_points_t, K, dist, joint_lengths, angle_lengths, point_lengths, 
             orig_img_sizes, joint_confidences) = batch

            image_sequences = image_sequences.to(local_rank, non_blocking=True)
            gt_heatmaps_t = gt_heatmaps_t.to(local_rank, non_blocking=True)
            gt_angles_t = gt_angles_t.to(local_rank, non_blocking=True)
            gt_angle_deltas = gt_angle_deltas.to(local_rank, non_blocking=True)
            gt_3d_t = gt_3d_points_t.to(local_rank, non_blocking=True)
            joint_confidences = joint_confidences.to(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred_heatmaps_t, pred_angles_t, pred_angle_deltas = model(image_sequences)

                # --- Loss Calculation ---
                # 1. Heatmap Loss (t)
                conf_mask_h = joint_confidences.unsqueeze(-1).unsqueeze(-1)
                loss_h = (loss_fn_h(pred_heatmaps_t, gt_heatmaps_t) * conf_mask_h).sum() / conf_mask_h.sum().clamp(min=1)

                # 2. Angle Loss (t)
                conf_mask_a = joint_confidences[:, :gt_angles_t.shape[1]]
                loss_a_t = (loss_fn_a(pred_angles_t, gt_angles_t) * conf_mask_a).sum() / conf_mask_a.sum().clamp(min=1)
                
                # 3. Angle Delta Loss (t -> t+1)
                loss_ad = (loss_fn_a(pred_angle_deltas, gt_angle_deltas) * conf_mask_a).sum() / conf_mask_a.sum().clamp(min=1)

            # 4. 3D Pose Loss (t) - calculated outside AMP context
            pred_heatmaps_np = pred_heatmaps_t.float().detach().cpu().numpy()
            pred_kpts_heatmap, pred_heatmap_conf = get_max_preds(pred_heatmaps_np)
            pred_visibility_mask = (torch.from_numpy(pred_heatmap_conf.squeeze(-1)).to(local_rank) >= HEATMAP_CONF_THRESHOLD).float()
            final_joint_conf = joint_confidences * pred_visibility_mask
            pred_visibility_np = final_joint_conf.detach().cpu().numpy() # Use combined confidence for PnP

            pred_3d_list = []
            for s in range(image_sequences.size(0)):
                robot = get_robot_kinematics(gt_class[s])
                joint_angles = robot._truncate_angles(pred_angles_t[s].detach().cpu().numpy())
                joint_coords_robot = robot.forward_kinematics(joint_angles)

                heatmap_size = (pred_heatmaps_t.shape[3], pred_heatmaps_t.shape[2])
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
                
                pred_3d_list.append(torch.tensor(joint_coords_cam, device=local_rank))

            padded_pred_3d_list = []
            max_len = gt_3d_t.shape[1]
            for p in pred_3d_list:
                pad_len = max_len - p.shape[0]
                if pad_len > 0:
                    padded_p = F.pad(p, (0, 0, 0, pad_len), "constant", 0)
                    padded_pred_3d_list.append(padded_p)
                else:
                    padded_pred_3d_list.append(p[:max_len])

            pred_3d = torch.stack(padded_pred_3d_list, dim=0)

            with torch.amp.autocast('cuda'):
                conf_mask_3d = final_joint_conf.unsqueeze(-1)
                loss_3d = (loss_fn_3D(pred_3d, gt_3d_t) * conf_mask_3d).sum() / conf_mask_3d.sum().clamp(min=1)

                total_loss = (weight_h * loss_h) + (weight_a_t * loss_a_t) + (weight_ad * loss_ad) + (weight_3d * loss_3d)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0: pbar.set_postfix(loss=total_loss.item())

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for batch in val_pbar:
                if batch is None: continue
                # ... (unpack batch similar to train loop) ...
                (image_sequences, gt_heatmaps_t, gt_angles_t, gt_angle_deltas, gt_class, 
                gt_3d_points_t, K, dist, joint_lengths, angle_lengths, point_lengths, 
                orig_img_sizes, joint_confidences) = batch

                image_sequences = image_sequences.to(local_rank, non_blocking=True)
                gt_heatmaps_t = gt_heatmaps_t.to(local_rank, non_blocking=True)
                gt_angles_t = gt_angles_t.to(local_rank, non_blocking=True)
                gt_angle_deltas = gt_angle_deltas.to(local_rank, non_blocking=True)
                gt_3d_t = gt_3d_points_t.to(local_rank, non_blocking=True)
                joint_confidences = joint_confidences.to(local_rank, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    pred_heatmaps_t, pred_angles_t, pred_angle_deltas = model(image_sequences)
                    
                    # --- Loss Calculation ---
                    conf_mask_h = joint_confidences.unsqueeze(-1).unsqueeze(-1)
                    loss_h = (loss_fn_h(pred_heatmaps_t, gt_heatmaps_t) * conf_mask_h).sum() / conf_mask_h.sum().clamp(min=1)
                    conf_mask_a = joint_confidences[:, :gt_angles_t.shape[1]]
                    loss_a_t = (loss_fn_a(pred_angles_t, gt_angles_t) * conf_mask_a).sum() / conf_mask_a.sum().clamp(min=1)
                    loss_ad = (loss_fn_a(pred_angle_deltas, gt_angle_deltas) * conf_mask_a).sum() / conf_mask_a.sum().clamp(min=1)
                    loss_3d = torch.tensor(0.0, device=local_rank) # Simplified for validation
                    total_loss = (weight_h * loss_h) + (weight_a_t * loss_a_t) + (weight_ad * loss_ad) + (weight_3d * loss_3d)

                loss_tensor = total_loss.detach()
                torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size
        
        scheduler.step()
        
        if rank == 0:
            avg_val_loss = val_loss / len(val_loader)
            wandb.log({
                "train_loss": total_loss.item(), # Note: This is last batch loss, not avg
                "val_loss": avg_val_loss,
                "loss_h": loss_h.item(),
                "loss_a_t": loss_a_t.item(),
                "loss_ad": loss_ad.item(),
                "loss_3d": loss_3d.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.6f}")
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"✨ New best model saved for '{ablation_mode}' with val_loss: {best_val_loss:.6f}")

            # ... (Saving logic remains same) ...
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            # Non-blocking save
            if save_thread is not None: save_thread.join()
            save_thread = threading.Thread(target=save_checkpoints, args=(checkpoint_data, model.module.state_dict(), CHECKPOINT_DIR, is_best))
            save_thread.start()

    if rank == 0:
        if save_thread is not None: save_thread.join()
        wandb.finish()
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINOv3 Motion Vector Pose Estimation Training")
    parser.add_argument('--ablation_mode', type=str, default='vit', help="Defines model variant and checkpoint paths.")
    args = parser.parse_args()
    main(args)