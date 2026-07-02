"""
Speed Benchmark: Depth Anything 3 vs DINOv3DepthEstimator
Measures inference speed, throughput, and GPU memory usage.
"""
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add Depth-Anything-3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Depth-Anything-3/src'))

from depth_model import DINOv3DepthEstimator
from depth_anything_3.api import DepthAnything3

def load_test_images(dataset_root, num_images=100):
    """Load random test images from dataset."""
    all_images = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))

    # Randomly sample
    import random
    random.seed(42)
    sampled = random.sample(all_images, min(num_images, len(all_images)))

    print(f"📁 Loaded {len(sampled)} test images")
    return sampled


def benchmark_depth_anything_3(image_paths, device='cuda', batch_size=1):
    """Benchmark Depth Anything 3 inference speed."""
    print("\n" + "="*80)
    print("🔵 Benchmarking Depth Anything 3 (Teacher Model)")
    print("="*80)

    # Load model using DepthAnything3 API
    model = DepthAnything3(model_name="da3nested-giant-large")
    model.model = model.model.to(device)
    model.device = device
    print(f"✓ Model loaded: Depth Anything 3 (DA3NESTED-GIANT-LARGE)")

    # Warmup
    print("🔥 Warming up...")
    dummy_img = Image.open(image_paths[0]).convert('RGB')
    dummy_img = dummy_img.resize((640, 360))
    for _ in range(10):
        _ = model.inference([dummy_img])

    torch.cuda.synchronize()

    # Single image benchmark
    print("\n📊 Single Image Inference:")
    torch.cuda.reset_peak_memory_stats(device)
    single_times = []

    for img_path in image_paths[:50]:  # Test 50 images
        img = Image.open(img_path).convert('RGB')
        img = img.resize((640, 360))

        torch.cuda.synchronize()
        start = time.time()

        prediction = model.inference([img])
        depth = prediction.depth[0]  # Get first depth map

        torch.cuda.synchronize()
        elapsed = time.time() - start
        single_times.append(elapsed)

    avg_time = np.mean(single_times)
    std_time = np.std(single_times)
    fps = 1.0 / avg_time

    print(f"  ⏱️  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  🚀 FPS: {fps:.2f}")
    print(f"  📏 Output shape: {depth.shape}")

    # GPU memory
    mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  💾 GPU Memory: {mem_allocated:.2f} GB")

    # Batch inference test (like dataset generation)
    print(f"\n📊 Batch Inference (batch_size={batch_size}):")
    batch_times = []

    for i in range(0, min(48, len(image_paths)), batch_size):
        batch_imgs = []
        for j in range(batch_size):
            if i + j < len(image_paths):
                img = Image.open(image_paths[i + j]).convert('RGB')
                img = img.resize((640, 360))
                batch_imgs.append(img)

        if len(batch_imgs) == 0:
            break

        torch.cuda.synchronize()
        start = time.time()

        prediction = model.inference(batch_imgs)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        batch_times.append(elapsed / len(batch_imgs))  # Per image time

    if batch_times:
        avg_batch_time = np.mean(batch_times)
        batch_fps = 1.0 / avg_batch_time
        print(f"  ⏱️  Average time per image: {avg_batch_time*1000:.2f} ms")
        print(f"  🚀 Batch FPS: {batch_fps:.2f}")
        print(f"  📈 Speedup vs single: {avg_time/avg_batch_time:.2f}x faster with batching")

    return {
        'model_name': 'Depth Anything 3',
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'gpu_memory_gb': mem_allocated,
        'output_shape': depth.shape,
        'batch_fps': batch_fps if batch_times else None
    }


def benchmark_dinov3_depth(image_paths, device='cuda', batch_size=1):
    """Benchmark DINOv3DepthEstimator inference speed."""
    print("\n" + "="*80)
    print("🟢 Benchmarking DINOv3DepthEstimator (Student Model)")
    print("="*80)

    # Load model
    model = DINOv3DepthEstimator(
        dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        depth_size=(280, 504)
    )

    # Try to load trained checkpoint
    checkpoint_paths = [
        "checkpoints_depth_depth_dinov3/best_model.pth",
        "checkpoints_depth_depth_dinov2_base_v1/best_model.pth"
    ]

    loaded = False
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint: {ckpt_path}")
            loaded = True
            break

    if not loaded:
        print("⚠️  No checkpoint found, using untrained model")

    model = model.to(device).eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    print("🔥 Warming up...")
    dummy_img = Image.open(image_paths[0]).convert('RGB')
    dummy_tensor = transform(dummy_img).unsqueeze(0).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_tensor)

    torch.cuda.synchronize()

    # Single image benchmark
    print("\n📊 Single Image Inference:")
    torch.cuda.reset_peak_memory_stats(device)
    single_times = []

    for img_path in image_paths[:50]:  # Test 50 images
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            depth = model(img_tensor)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        single_times.append(elapsed)

    avg_time = np.mean(single_times)
    std_time = np.std(single_times)
    fps = 1.0 / avg_time

    print(f"  ⏱️  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  🚀 FPS: {fps:.2f}")
    print(f"  📏 Output shape: {tuple(depth.shape)}")

    # GPU memory
    mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  💾 GPU Memory: {mem_allocated:.2f} GB")

    # Batch inference test
    print(f"\n📊 Batch Inference (batch_size={batch_size}):")
    batch_times = []

    for i in range(0, min(50, len(image_paths)), batch_size):
        batch_imgs = []
        for j in range(batch_size):
            if i + j < len(image_paths):
                img = Image.open(image_paths[i + j]).convert('RGB')
                batch_imgs.append(transform(img))

        if len(batch_imgs) == 0:
            break

        batch_tensor = torch.stack(batch_imgs).to(device)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            depths = model(batch_tensor)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        batch_times.append(elapsed / len(batch_imgs))  # Per image time

    if batch_times:
        avg_batch_time = np.mean(batch_times)
        batch_fps = 1.0 / avg_batch_time
        print(f"  ⏱️  Average time per image: {avg_batch_time*1000:.2f} ms")
        print(f"  🚀 Batch FPS: {batch_fps:.2f}")
        print(f"  📈 Speedup vs single: {fps/batch_fps:.2f}x slower (overhead from batching)")

    return {
        'model_name': 'DINOv3DepthEstimator',
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'gpu_memory_gb': mem_allocated,
        'output_shape': tuple(depth.shape),
        'batch_fps': batch_fps if batch_times else None
    }


def main():
    print("🏁 Depth Estimation Speed Benchmark")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Load test images
    dataset_root = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset"
    image_paths = load_test_images(dataset_root, num_images=100)

    # Benchmark Depth Anything 3 (with batch inference like dataset generation)
    da3_results = benchmark_depth_anything_3(image_paths, device, batch_size=16)

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark DINOv3DepthEstimator
    dinov3_results = benchmark_dinov3_depth(image_paths, device, batch_size=8)

    # Summary comparison
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)

    print(f"\n{'Model':<30} {'Avg Time (ms)':<15} {'FPS':<10} {'Memory (GB)':<15}")
    print("-" * 80)
    print(f"{da3_results['model_name']:<30} {da3_results['avg_time_ms']:<15.2f} {da3_results['fps']:<10.2f} {da3_results['gpu_memory_gb']:<15.2f}")
    print(f"{dinov3_results['model_name']:<30} {dinov3_results['avg_time_ms']:<15.2f} {dinov3_results['fps']:<10.2f} {dinov3_results['gpu_memory_gb']:<15.2f}")

    # Speed comparison
    speedup = da3_results['avg_time_ms'] / dinov3_results['avg_time_ms']
    print("\n🏆 Winner:")
    if speedup > 1.0:
        print(f"   DINOv3DepthEstimator is {speedup:.2f}x FASTER than Depth Anything 3")
    else:
        print(f"   Depth Anything 3 is {1/speedup:.2f}x FASTER than DINOv3DepthEstimator")

    print("\n💡 Recommendation:")
    if da3_results['fps'] > 30:
        print("   ✅ Depth Anything 3 is fast enough (>30 FPS)")
        print("   → Consider using teacher model directly (no training needed)")
    elif speedup > 2.0:
        print("   ✅ DINOv3DepthEstimator is significantly faster")
        print("   → Training student model is worthwhile for speed")
    else:
        print("   ⚠️  Speed difference is not significant")
        print("   → Consider using teacher model for better quality")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
