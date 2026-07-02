"""
Generate depth dataset from original dataset using Depth Anything 3 with Multi-GPU support.
Uses multiprocessing to parallelize across multiple GPUs.
"""
import os
import glob
import torch
import numpy as np
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from depth_anything_3.api import DepthAnything3


def find_all_images(root_dir, extensions=['.jpg', '.jpeg', '.png']):
    """
    Recursively find all image files in the directory.

    Args:
        root_dir: Root directory to search
        extensions: List of image extensions to look for

    Returns:
        List of tuples (absolute_path, relative_path)
    """
    root_path = Path(root_dir)
    image_files = []

    for ext in extensions:
        for img_path in root_path.rglob(f'*{ext}'):
            if img_path.is_file():
                rel_path = img_path.relative_to(root_path)
                image_files.append((str(img_path), str(rel_path)))

    return sorted(image_files)


def resize_image(image_path, target_size=(640, 360)):
    """
    Load and resize image to target size.

    Args:
        image_path: Path to the image
        target_size: (width, height) tuple

    Returns:
        PIL Image
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    return img_resized


def process_batch(model, image_paths, target_size=(640, 360), temp_dir="/tmp/depth_temp"):
    """
    Process a batch of images with Depth Anything 3.

    Args:
        model: DepthAnything3 model
        image_paths: List of image paths
        target_size: (width, height) tuple for resizing
        temp_dir: Temporary directory for resized images

    Returns:
        prediction object from model.inference()
    """
    # Resize all images in batch
    resized_images = [resize_image(img_path, target_size) for img_path in image_paths]

    # Save temporary resized images for model inference
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)

    temp_paths = []
    for i, img in enumerate(resized_images):
        temp_file = temp_path / f"temp_{os.getpid()}_{i}.png"
        img.save(temp_file)
        temp_paths.append(str(temp_file))

    # Run inference
    prediction = model.inference(temp_paths)

    # Clean up temp files
    for temp_file in temp_paths:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return prediction


def save_depth(depth_array, output_path):
    """
    Save depth array as .npy file.

    Args:
        depth_array: (H, W) numpy array
        output_path: Path to save the depth file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, depth_array)


def worker_process(gpu_id, image_subset, source_root, target_root, target_size, batch_size, progress_queue):
    """
    Worker process that processes images on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        image_subset: List of (abs_path, rel_path) tuples to process
        source_root: Source dataset root directory
        target_root: Target output root directory
        target_size: (width, height) for resizing
        batch_size: Number of images to process per batch
        progress_queue: Queue for progress reporting
    """
    try:
        # Set CUDA device for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device("cuda:0")  # Always use device 0 since CUDA_VISIBLE_DEVICES is set

        # Initialize model
        model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
        model = model.to(device=device)

        # Create temp directory for this worker
        temp_dir = f"/tmp/depth_temp_gpu{gpu_id}"

        # Process in batches
        num_batches = (len(image_subset) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_subset))
            batch_files = image_subset[start_idx:end_idx]

            # Extract paths and relative paths
            abs_paths = [f[0] for f in batch_files]
            rel_paths = [f[1] for f in batch_files]

            try:
                # Run inference
                prediction = process_batch(model, abs_paths, target_size, temp_dir)

                # Save depth maps
                for i, rel_path in enumerate(rel_paths):
                    # Change extension to .npy
                    depth_rel_path = Path(rel_path).with_suffix('.npy')
                    depth_output_path = Path(target_root) / depth_rel_path

                    # Save depth array
                    save_depth(prediction.depth[i], depth_output_path)

                # Report progress
                progress_queue.put(len(batch_files))

            except Exception as e:
                progress_queue.put(('error', gpu_id, batch_idx, str(e)))
                continue

        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        progress_queue.put(('done', gpu_id))

    except Exception as e:
        progress_queue.put(('fatal', gpu_id, str(e)))


def main():
    # Configuration
    source_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
    target_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/depth_dataset"
    target_size = (640, 360)  # (width, height)
    batch_size = 16  # Process 16 images at a time per GPU
    num_gpus = torch.cuda.device_count()

    print("=" * 80)
    print("Multi-GPU Depth Dataset Generation")
    print("=" * 80)
    print(f"Source directory: {source_root}")
    print(f"Target directory: {target_root}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Number of GPUs: {num_gpus}")
    print("=" * 80)

    if num_gpus == 0:
        print("❌ No GPUs available! Exiting.")
        return

    # Find all images
    print("\nScanning for images...")
    image_files = find_all_images(source_root)
    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        print("No images found! Exiting.")
        return

    # Split images across GPUs
    images_per_gpu = len(image_files) // num_gpus
    gpu_subsets = []

    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        if i == num_gpus - 1:
            # Last GPU gets any remaining images
            end_idx = len(image_files)
        else:
            end_idx = (i + 1) * images_per_gpu

        gpu_subsets.append(image_files[start_idx:end_idx])
        print(f"GPU {i}: {len(gpu_subsets[i])} images ({start_idx} to {end_idx-1})")

    print("\n" + "=" * 80)
    print("Starting multi-GPU processing...")
    print("=" * 80)

    # Create progress queue
    progress_queue = mp.Queue()

    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_subsets[gpu_id], source_root, target_root, target_size, batch_size, progress_queue)
        )
        p.start()
        processes.append(p)

    # Monitor progress
    total_processed = 0
    active_gpus = num_gpus
    pbar = tqdm(total=len(image_files), desc="Total Progress", unit="img")

    while active_gpus > 0:
        msg = progress_queue.get()

        if isinstance(msg, int):
            # Progress update
            total_processed += msg
            pbar.update(msg)

        elif isinstance(msg, tuple):
            if msg[0] == 'done':
                gpu_id = msg[1]
                print(f"\n✓ GPU {gpu_id} finished")
                active_gpus -= 1

            elif msg[0] == 'error':
                gpu_id, batch_idx, error = msg[1], msg[2], msg[3]
                print(f"\n⚠ GPU {gpu_id} error in batch {batch_idx}: {error}")

            elif msg[0] == 'fatal':
                gpu_id, error = msg[1], msg[2]
                print(f"\n❌ GPU {gpu_id} fatal error: {error}")
                active_gpus -= 1

    pbar.close()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("\n" + "=" * 80)
    print("Multi-GPU depth dataset generation completed!")
    print(f"Output directory: {target_root}")
    print(f"Total images processed: {total_processed}")
    print("=" * 80)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
