"""
Training script for DINOv3DepthEstimator.
Knowledge distillation from Depth Anything 3 (teacher model).
"""
import os
import random
import wandb
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

from depth_dataset import DepthDataset, depth_collate_fn
from depth_model import DINOv3DepthEstimator
from utils import setup_ddp, cleanup_ddp, save_checkpoints


# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DepthLoss(nn.Module):
    """
    Combined depth loss with multiple components:
    - L1 loss: robust to outliers
    - Gradient loss: preserves edges and details
    """
    def __init__(self, l1_weight=1.0, grad_weight=0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight

    def gradient_loss(self, pred, gt):
        """
        Gradient loss to preserve depth discontinuities and edges.
        """
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        # L1 loss on gradients
        loss_dx = F.l1_loss(pred_dx, gt_dx)
        loss_dy = F.l1_loss(pred_dy, gt_dy)

        return (loss_dx + loss_dy) / 2.0

    def forward(self, pred_depth, gt_depth):
        """
        Args:
            pred_depth: (B, 1, H, W) - predicted depth
            gt_depth: (B, 1, H, W) - ground truth depth

        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # Resize prediction to match GT size if needed
        if pred_depth.shape != gt_depth.shape:
            pred_depth = F.interpolate(
                pred_depth,
                size=gt_depth.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # L1 loss (robust to outliers)
        l1_loss = F.l1_loss(pred_depth, gt_depth)

        # Gradient loss (preserve edges)
        grad_loss = self.gradient_loss(pred_depth, gt_depth)

        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.grad_weight * grad_loss

        loss_dict = {
            'l1_loss': l1_loss,
            'grad_loss': grad_loss,
            'total_loss': total_loss
        }

        return total_loss, loss_dict


def main(args):
    rank, local_rank, world_size = setup_ddp()
    save_thread = None

    # Training hyperparameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size  # Per-GPU batch size
    EPOCHS = args.epochs
    VAL_RATIO = 0.1
    NUM_WORKERS = 4

    # Model configuration
    MODEL_NAME = args.model_name
    DEPTH_SIZE = tuple(map(int, args.depth_size.split(',')))  # e.g., "280,504"

    # Paths
    RGB_ROOT = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
    DEPTH_ROOT = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/depth_dataset"

    WANDB_PROJECT = "DINOv3_Depth_Estimation"
    CHECKPOINT_DIR = f"checkpoints_depth_{args.run_name}"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

    start_epoch = 0
    best_val_loss = float('inf')

    # Initialize model
    model = DINOv3DepthEstimator(
        dino_model_name=MODEL_NAME,
        depth_size=DEPTH_SIZE
    )
    model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        static_graph=True
    )

    # Loss function
    criterion = DepthLoss(l1_weight=1.0, grad_weight=0.5)

    # Optimizer and scheduler
    effective_lr = LEARNING_RATE * (world_size ** 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if exists
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
            print(f"✅ Checkpoint loaded from {LATEST_CHECKPOINT_PATH}")
            print(f"   Resuming from epoch {start_epoch}, Best Val Loss: {best_val_loss:.6f}")

    elif os.path.exists(CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.module.load_state_dict(state_dict)
        if rank == 0:
            print(f"✅ Model weights loaded from {CHECKPOINT_PATH}")
    else:
        if rank == 0:
            print(f"ℹ️ No checkpoint found, training from scratch")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((640, 360)),  # Match training data preprocessing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    if rank == 0:
        print("Loading depth dataset...")

    full_dataset = DepthDataset(RGB_ROOT, DEPTH_ROOT, transform=transform)

    if len(full_dataset) < 2:
        raise RuntimeError("Dataset must contain at least two samples to create train/val splits.")

    # Train/val split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * VAL_RATIO)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=depth_collate_fn,
        sampler=train_sampler,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=depth_collate_fn,
        sampler=val_sampler,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Weights & Biases logging
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(
            project=WANDB_PROJECT,
            name=args.run_name,
            config={
                "base_learning_rate": LEARNING_RATE,
                "effective_learning_rate": effective_lr,
                "per_gpu_batch_size": BATCH_SIZE,
                "total_batch_size": BATCH_SIZE * world_size,
                "epochs": EPOCHS,
                "world_size": world_size,
                "model_name": MODEL_NAME,
                "depth_size": DEPTH_SIZE,
            },
            resume="allow"
        )

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for images, gt_depths in pbar:
            images = images.to(local_rank)
            gt_depths = gt_depths.to(local_rank)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with torch.amp.autocast('cuda'):
                pred_depths = model(images)
                loss, loss_dict = criterion(pred_depths, gt_depths)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss
            loss_tensor = loss.detach().clone()
            torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
            train_loss += loss_tensor.item() / world_size

            if rank == 0:
                pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for images, gt_depths in val_pbar:
                images = images.to(local_rank)
                gt_depths = gt_depths.to(local_rank)

                with torch.amp.autocast('cuda'):
                    pred_depths = model(images)
                    loss, loss_dict = criterion(pred_depths, gt_depths)

                loss_tensor = loss.detach().clone()
                torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)
                val_loss += loss_tensor.item() / world_size

                if rank == 0:
                    val_pbar.set_postfix(loss=loss_tensor.item() / world_size)

        scheduler.step()

        # Logging and checkpointing
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "l1_loss": loss_dict['l1_loss'].item(),
                "grad_loss": loss_dict['grad_loss'].item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"✨ New best model saved with val_loss: {best_val_loss:.6f}")

            # Save checkpoint
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
    parser = argparse.ArgumentParser(description="DINOv3 Depth Estimation Training")
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-base',
                       help='DINOv2/v3 model name')
    parser.add_argument('--depth_size', type=str, default='280,504',
                       help='Depth map output size (H,W)')
    parser.add_argument('--run_name', type=str, default='depth_v1',
                       help='Run name for wandb and checkpoints')

    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    main(args)
