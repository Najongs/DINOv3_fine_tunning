import os
import random
import argparse
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from human_pose_dataset import (
    COCOHumanPoseDataset,
    KeypointOcclusionAugmentor,
    human_pose_collate_fn,
)
from human_pose_model import DINOv3HumanPoseEstimator
from utils import setup_ddp, cleanup_ddp

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def weighted_mse_loss(pred_heatmaps, gt_heatmaps, weights):
    """
    Compute weighted MSE loss for heatmaps.

    Args:
        pred_heatmaps: (B, K, H, W) predicted heatmaps
        gt_heatmaps: (B, K, H, W) ground truth heatmaps
        weights: (B, K) per-keypoint weights

    Returns:
        loss: scalar weighted MSE loss
    """
    # Compute per-pixel MSE
    mse = (pred_heatmaps - gt_heatmaps) ** 2  # (B, K, H, W)

    # Average over spatial dimensions
    mse = mse.mean(dim=[2, 3])  # (B, K)

    # Apply per-keypoint weights
    weighted_mse = mse * weights  # (B, K)

    # Average over batch and keypoints
    loss = weighted_mse.sum() / (weights.sum() + 1e-6)

    return loss


def train_one_epoch(model, train_loader, optimizer, scaler, epoch, rank, local_rank):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=(rank != 0))
    for images, gt_heatmaps, keypoint_weights in pbar:
        images = images.to(local_rank)
        gt_heatmaps = gt_heatmaps.to(local_rank)
        keypoint_weights = keypoint_weights.to(local_rank)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            pred_heatmaps = model(images)
            loss = weighted_mse_loss(pred_heatmaps, gt_heatmaps, keypoint_weights)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, epoch, rank, local_rank):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=(rank != 0))
    with torch.no_grad():
        for images, gt_heatmaps, keypoint_weights in pbar:
            images = images.to(local_rank)
            gt_heatmaps = gt_heatmaps.to(local_rank)
            keypoint_weights = keypoint_weights.to(local_rank)

            # Forward pass
            with torch.amp.autocast('cuda'):
                pred_heatmaps = model(images)
                loss = weighted_mse_loss(pred_heatmaps, gt_heatmaps, keypoint_weights)

            total_loss += loss.item()
            num_batches += 1

            if rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, checkpoint_path):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def main(args):
    # Setup DDP
    rank, local_rank, world_size = setup_ddp()

    # Training hyperparameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size  # Per-GPU batch size
    EPOCHS = args.epochs
    VAL_RATIO = 0.1
    NUM_WORKERS = args.num_workers
    IMAGE_SIZE = args.image_size
    HEATMAP_SIZE = args.heatmap_size

    # Paths
    TRAIN_IMAGE_DIR = args.train_image_dir
    TRAIN_ANNOTATION_FILE = args.train_annotation_file
    VAL_IMAGE_DIR = args.val_image_dir if args.val_image_dir else TRAIN_IMAGE_DIR
    VAL_ANNOTATION_FILE = args.val_annotation_file if args.val_annotation_file else TRAIN_ANNOTATION_FILE

    CHECKPOINT_DIR = args.checkpoint_dir
    BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

    # Model selection
    MODEL_NAME = args.model_name

    if rank == 0:
        print(f"Training configuration:")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Image size: {IMAGE_SIZE}")
        print(f"  Heatmap size: {HEATMAP_SIZE}")
        print(f"  Batch size per GPU: {BATCH_SIZE}")
        print(f"  Total batch size: {BATCH_SIZE * world_size}")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  Number of workers: {NUM_WORKERS}")

    # Initialize model
    model = DINOv3HumanPoseEstimator(
        dino_model_name=MODEL_NAME,
        heatmap_size=HEATMAP_SIZE,
        num_keypoints=17
    )
    model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        static_graph=True
    )

    # Optimizer, scheduler, scaler
    effective_lr = LEARNING_RATE * (world_size ** 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(LATEST_CHECKPOINT_PATH):
        if rank == 0:
            print(f"Loading checkpoint from {LATEST_CHECKPOINT_PATH}...")

        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=map_location)

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Occlusion augmentation for training
    occlusion_augmentor = KeypointOcclusionAugmentor(
        prob=0.2,
        min_occlusions=1,
        max_occlusions=4,
        min_patch_ratio=0.05,
        max_patch_ratio=0.15,
    )

    # Datasets
    if rank == 0:
        print("Loading datasets...")

    train_dataset = COCOHumanPoseDataset(
        image_dir=TRAIN_IMAGE_DIR,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transform=transform,
        heatmap_size=HEATMAP_SIZE,
        sigma=3.0,
        occlusion_augmentor=occlusion_augmentor
    )

    # Use separate validation set if provided, otherwise use train set
    if args.val_annotation_file:
        val_dataset = COCOHumanPoseDataset(
            image_dir=VAL_IMAGE_DIR,
            annotation_file=VAL_ANNOTATION_FILE,
            transform=transform,
            heatmap_size=HEATMAP_SIZE,
            sigma=3.0,
            occlusion_augmentor=None  # No augmentation for validation
        )
    else:
        # Split train dataset into train/val
        if rank == 0:
            print("No separate validation set provided, splitting train set...")
        total_size = len(train_dataset)
        val_size = int(total_size * VAL_RATIO)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

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
        collate_fn=human_pose_collate_fn,
        sampler=train_sampler,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=human_pose_collate_fn,
        sampler=val_sampler,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Initialize wandb
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": MODEL_NAME,
                "base_learning_rate": LEARNING_RATE,
                "effective_learning_rate": effective_lr,
                "per_gpu_batch_size": BATCH_SIZE,
                "total_batch_size": BATCH_SIZE * world_size,
                "epochs": EPOCHS,
                "world_size": world_size,
                "num_workers": NUM_WORKERS,
                "image_size": IMAGE_SIZE,
                "heatmap_size": HEATMAP_SIZE,
            },
            resume="allow"
        )

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, rank, local_rank)

        # Validate
        val_loss = validate(model, val_loader, epoch, rank, local_rank)

        # Step scheduler
        scheduler.step()

        # Log to wandb
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {current_lr:.2e}")

            # Save latest checkpoint
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, LATEST_CHECKPOINT_PATH)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, BEST_CHECKPOINT_PATH)
                print(f"  ✅ New best model saved! Val Loss: {best_val_loss:.6f}")

    # Cleanup
    if rank == 0:
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Human Pose Estimation Model")

    # Dataset paths
    parser.add_argument("--train_image_dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--train_annotation_file", type=str, required=True, help="Path to training COCO annotation file")
    parser.add_argument("--val_image_dir", type=str, default=None, help="Path to validation images directory (optional)")
    parser.add_argument("--val_annotation_file", type=str, default=None, help="Path to validation COCO annotation file (optional)")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", help="DINOv3 model name")
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Input image size (H, W)")
    parser.add_argument("--heatmap_size", type=int, nargs=2, default=[512, 512], help="Output heatmap size (H, W)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers per GPU")

    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_human_pose", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="DINOv3_HumanPose", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="human_pose_training", help="Wandb run name")

    args = parser.parse_args()

    main(args)
