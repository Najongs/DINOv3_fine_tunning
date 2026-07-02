#!/usr/bin/env python3
"""
Test MediaPipe Hands detection speed
"""
import time
import cv2
import mediapipe as mp
from pathlib import Path

# Test image
test_image = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Fr5/Fr5_7th_250526/left/zed_38007749_left_1748249364.809.jpg"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load image
image = cv2.imread(test_image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("=" * 60)
print("MediaPipe Hands Speed Test")
print("=" * 60)

# Warmup
print("\nWarmup (3 iterations)...")
for i in range(3):
    start = time.time()
    results = hands_detector.process(image_rgb)
    elapsed = time.time() - start
    print(f"  Warmup {i+1}: {elapsed:.4f}s")

# Test
print("\nTest (10 iterations)...")
timings = []
for i in range(10):
    start = time.time()
    results = hands_detector.process(image_rgb)
    elapsed = time.time() - start
    timings.append(elapsed)

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    print(f"  Test {i+1}: {elapsed:.4f}s - Detected {num_hands} hands")

print("\n" + "=" * 60)
print(f"Average: {sum(timings)/len(timings):.4f}s")
print(f"Min: {min(timings):.4f}s")
print(f"Max: {max(timings):.4f}s")
print(f"FPS: {1.0/(sum(timings)/len(timings)):.2f}")
print("=" * 60)

hands_detector.close()
