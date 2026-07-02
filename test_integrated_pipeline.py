#!/usr/bin/env python3
"""
Test script for integrated pipeline with warmup and multiple images
- Warmup: 2-3 iterations
- Test: 5 images with individual visualization
"""
import os
import argparse
import time
from pathlib import Path
from integrated_pipeline import IntegratedPipeline, visualize_results


def main(args):
    # Initialize pipeline
    print("=" * 80)
    print("Initializing Pipeline...")
    print("=" * 80)

    pipeline = IntegratedPipeline(
        robot_checkpoint=args.robot_checkpoint,
        robot_model_name=args.robot_model_name,
        robot_heatmap_size=tuple(map(int, args.robot_heatmap_size.split(','))),
        depth_model_name=args.depth_model_name,
        yolo_pose_model=args.yolo_pose_model,
        use_multi_gpu=args.use_multi_gpu,
        robot_gpu=args.robot_gpu,
        depth_gpu=args.depth_gpu,
        human_gpu=args.human_gpu
    )

    # Get test images
    test_images = []
    if args.image_dir:
        # Use all images from directory
        image_dir = Path(args.image_dir)
        image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return
        # Take first 5 for testing
        test_images = image_paths[:5]
    elif args.image_list:
        try:
            with open(args.image_list, 'r') as f:
                test_images = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Image list file not found at {args.image_list}")
            return
    elif args.image_path:
        # Use single image repeated
        test_images = [args.image_path] * 5

    print(f"\nTest images: {len(test_images)}")
    for i, img in enumerate(test_images):
        print(f"  {i+1}. {img}")

    # Warmup
    print("\n" + "=" * 80)
    print("WARMUP Phase (3 iterations)")
    print("=" * 80)

    warmup_iterations = 3
    warmup_timings = []

    for i in range(warmup_iterations):
        print(f"\nWarmup {i+1}/{warmup_iterations}...")
        start = time.time()
        _ = pipeline.predict(str(test_images[0]), robot_class=args.robot_class)
        elapsed = time.time() - start
        warmup_timings.append(elapsed)
        print(f"  Time: {elapsed:.3f}s")

    print(f"\nWarmup complete. Average: {sum(warmup_timings)/len(warmup_timings):.3f}s")

    # Test phase
    print("\n" + "=" * 80)
    print("TEST Phase (5 images)")
    print("=" * 80)

    test_timings = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, image_path in enumerate(test_images):
        print(f"\n[{i+1}/5] Processing: {Path(image_path).name}")
        print("-" * 80)

        start = time.time()
        results = pipeline.predict(str(image_path), robot_class=args.robot_class)
        elapsed = time.time() - start
        test_timings.append(elapsed)

        # Print timings
        if results.get('timings'):
            timings = results['timings']
            print(f"  Robot Pose:  {timings.get('robot', 0):.3f}s")
            print(f"  Depth:       {timings.get('depth', 0):.3f}s")
            print(f"  Human Pose:  {timings.get('human', 0):.3f}s")
            print(f"  Hand Detect: {timings.get('hands', 0):.3f}s")
            print(f"  Total:       {timings.get('total', 0):.3f}s")

        # Save visualization
        output_path = output_dir / f"result_{i+1:02d}_{Path(image_path).stem}.png"
        visualize_results(str(image_path), results, str(output_path))
        print(f"  ✓ Saved: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"  Average time: {sum(warmup_timings)/len(warmup_timings):.3f}s")
    print(f"  Min: {min(warmup_timings):.3f}s, Max: {max(warmup_timings):.3f}s")
    print()
    print(f"Test iterations: {len(test_timings)}")
    print(f"  Average time: {sum(test_timings)/len(test_timings):.3f}s")
    print(f"  Min: {min(test_timings):.3f}s, Max: {max(test_timings):.3f}s")
    print(f"  FPS: {len(test_timings)/sum(test_timings):.2f}")
    print()
    print(f"Individual timings:")
    for i, t in enumerate(test_timings):
        print(f"  Image {i+1}: {t:.3f}s")
    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Integrated Pipeline with Warmup")

    # Robot model
    parser.add_argument("--robot_checkpoint", type=str, required=True)
    parser.add_argument("--robot_model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--robot_heatmap_size", type=str, default="512,512")
    parser.add_argument("--robot_class", type=str, default="Research3")

    # Depth model
    parser.add_argument("--depth_model_name", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")

    # YOLO-Pose
    parser.add_argument("--yolo_pose_model", type=str, default="yolov8l-pose.pt")

    # GPU settings
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--robot_gpu", type=int, default=0)
    parser.add_argument("--depth_gpu", type=int, default=1)
    parser.add_argument("--human_gpu", type=int, default=2)

    # Input/Output
    parser.add_argument("--image_path", type=str, help="Single image path (for testing)")
    parser.add_argument("--image_dir", type=str, help="Directory with multiple images (takes first 5)")
    parser.add_argument("--image_list", type=str, help="A file containing a list of image paths to process.")
    parser.add_argument("--output_dir", type=str, default="integrated_test_results",
                       help="Output directory for visualizations")

    args = parser.parse_args()

    if not args.image_path and not args.image_dir and not args.image_list:
        parser.error("Must provide either --image_path, --image_dir, or --image_list")

    main(args)
