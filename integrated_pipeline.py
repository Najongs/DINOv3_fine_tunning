"""
Integrated Multi-Model Pipeline for Robot Scene Understanding
- Robot Pose: Custom DINOv3-based model
- Depth: Depth Anything 3
- Human Pose: RTMPose

Supports both single-GPU (sequential) and multi-GPU (parallel) execution.
"""
import os
import time
import argparse
import threading
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import custom robot pose model
from model import DINOv3PoseEstimator
from kinematics import get_robot_kinematics
from dataset import _scale_points, HEATMAP_SIZE

# Import Depth Anything 3
try:
    from depth_anything_3.api import DepthAnything3
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False
    print("⚠ Depth Anything 3 not available. Install with: pip install depth-anything-v2")

# Import YOLO-Pose (Ultralytics)
try:
    from ultralytics import YOLO
    YOLOPOSE_AVAILABLE = True
except ImportError:
    YOLOPOSE_AVAILABLE = False
    print("⚠ Ultralytics YOLO-Pose not available. Install with: pip install ultralytics")

# Import MediaPipe Hands
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available. Install with: pip install mediapipe")


class IntegratedPipeline:
    """
    Integrated pipeline combining:
    1. Robot Pose Estimation (custom DINOv3 model)
    2. Depth Estimation (Depth Anything 3)
    3. Human Pose Estimation (YOLO-Pose)
    4. Hand Keypoint Detection (MediaPipe Hands)
    """
    def __init__(
        self,
        robot_checkpoint,
        robot_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        robot_heatmap_size=(640, 360),
        depth_model_name="depth-anything/DA3NESTED-GIANT-LARGE",
        yolo_pose_model="yolov8l-pose.pt",
        use_mediapipe_hands=True,
        use_multi_gpu=True,
        robot_gpu=0,
        depth_gpu=1,
        human_gpu=2
    ):
        """
        Initialize the integrated pipeline.

        Args:
            robot_checkpoint: Path to robot pose model checkpoint
            robot_model_name: DINOv3 model name
            robot_heatmap_size: (H, W) heatmap size for robot model
            depth_model_name: Depth Anything model name
            rtm_det_config: RTMDet config file path
            rtm_det_checkpoint: RTMDet checkpoint path
            rtm_pose_config: RTMPose config file path
            rtm_pose_checkpoint: RTMPose checkpoint path
            use_multi_gpu: Whether to use multiple GPUs for parallel execution
            robot_gpu: GPU ID for robot model
            depth_gpu: GPU ID for depth model
            human_gpu: GPU ID for human pose model
        """
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() >= 3
        self.robot_gpu = robot_gpu
        self.depth_gpu = depth_gpu
        self.human_gpu = human_gpu

        print("=" * 80)
        print("Initializing Integrated Pipeline")
        print("=" * 80)
        print(f"Multi-GPU mode: {self.use_multi_gpu}")
        if self.use_multi_gpu:
            print(f"  Robot model on GPU {robot_gpu}")
            print(f"  Depth model on GPU {depth_gpu}")
            print(f"  Human model on GPU {human_gpu}")

        # 1. Initialize Robot Pose Model
        print("\n[1/3] Loading Robot Pose Model...")
        self.robot_device = torch.device(f'cuda:{robot_gpu}')
        self.robot_model = DINOv3PoseEstimator(
            dino_model_name=robot_model_name,
            heatmap_size=robot_heatmap_size
        ).to(self.robot_device)

        # Load checkpoint
        checkpoint = torch.load(robot_checkpoint, map_location=self.robot_device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            trainable_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('backbone.model.')}
            self.robot_model.load_state_dict(trainable_state_dict, strict=False)
        else:
            trainable_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('backbone.model.')}
            self.robot_model.load_state_dict(trainable_state_dict, strict=False)

        self.robot_model.eval()
        self.robot_heatmap_size = robot_heatmap_size

        # Robot transform
        self.robot_transform = transforms.Compose([
            transforms.Resize(robot_heatmap_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("✓ Robot Pose Model loaded")

        # 2. Initialize Depth Model
        self.depth_model = None
        if DEPTH_AVAILABLE:
            print("\n[2/3] Loading Depth Model...")
            if self.use_multi_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(depth_gpu)
                self.depth_model = DepthAnything3.from_pretrained(depth_model_name)
                self.depth_model = self.depth_model.to(device=f'cuda:0')  # Will be cuda:1 in the main env
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                self.depth_model = DepthAnything3.from_pretrained(depth_model_name)
                self.depth_model = self.depth_model.to(device=f'cuda:{depth_gpu}')
            print("✓ Depth Model loaded")
        else:
            print("\n[2/3] Depth Model not available (skipping)")

        # 3. Initialize YOLO-Pose
        self.yolo_model = None
        if YOLOPOSE_AVAILABLE:
            print("\n[3/4] Loading YOLO-Pose Model...")
            self.yolo_model = YOLO(yolo_pose_model)
            if self.use_multi_gpu:
                self.yolo_model.to(f'cuda:{human_gpu}')
            print("✓ YOLO-Pose Model loaded")
        else:
            print("\n[3/4] YOLO-Pose not available (skipping)")

        # 4. Initialize MediaPipe Hands
        self.hands_detector = None
        self.use_mediapipe_hands = use_mediapipe_hands and MEDIAPIPE_AVAILABLE
        if self.use_mediapipe_hands:
            print("\n[4/4] Loading MediaPipe Hands...")
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe Hands loaded")
        else:
            print("\n[4/4] MediaPipe Hands not available (skipping)")

        print("\n" + "=" * 80)
        print("✓ Pipeline initialized successfully!")
        print("=" * 80)

    @torch.no_grad()
    def predict_robot_pose(self, image_pil, robot_class='Research3'):
        """
        Predict robot pose (heatmaps + joint angles).

        Args:
            image_pil: PIL Image
            robot_class: Robot type ('Research3', 'Fr5', etc.)

        Returns:
            dict with keys: 'heatmaps', 'angles', 'keypoints_2d', '3d_points'
        """
        # Preprocess
        img_tensor = self.robot_transform(image_pil).unsqueeze(0).to(self.robot_device)

        # Inference
        pred_heatmaps, pred_angles = self.robot_model(img_tensor)

        # Extract keypoints from heatmaps
        heatmaps_np = pred_heatmaps[0].cpu().numpy()  # (K, H, W)
        num_joints = heatmaps_np.shape[0]

        keypoints_heatmap = []
        for j in range(num_joints):
            hm = heatmaps_np[j]
            y_max, x_max = np.unravel_index(np.argmax(hm), hm.shape)
            keypoints_heatmap.append([x_max, y_max])
        keypoints_heatmap = np.array(keypoints_heatmap, dtype=np.float32)

        # Scale to original image size
        orig_w, orig_h = image_pil.size
        keypoints_2d = _scale_points(
            keypoints_heatmap,
            from_size=(self.robot_heatmap_size[1], self.robot_heatmap_size[0]),
            to_size=(orig_w, orig_h)
        )

        # Forward kinematics for 3D points
        angles_np = pred_angles[0].cpu().numpy()
        robot = get_robot_kinematics(robot_class)
        angles_truncated = robot._truncate_angles(angles_np)
        points_3d = robot.forward_kinematics(angles_truncated)

        return {
            'heatmaps': heatmaps_np,
            'angles': angles_np,
            'keypoints_2d': keypoints_2d,
            '3d_points': points_3d
        }

    def predict_depth(self, image_path):
        """
        Predict depth map.

        Args:
            image_path: Path to image file

        Returns:
            depth map (H, W) numpy array
        """
        if self.depth_model is None:
            return None

        prediction = self.depth_model.inference([image_path])
        return prediction.depth[0]

    def predict_human_pose(self, image_path):
        """
        Predict human pose using YOLO-Pose.

        Args:
            image_path: Path to image file

        Returns:
            YOLO pose results
        """
        if self.yolo_model is None:
            return None

        # YOLO-Pose inference (detection + pose in one step)
        results = self.yolo_model(image_path, verbose=False)

        return results[0] if len(results) > 0 else None

    def predict_hands(self, image_path):
        """
        Detect hand keypoints using MediaPipe Hands.

        Args:
            image_path: Path to image file

        Returns:
            dict with 'hands': list of hand landmarks (21 keypoints per hand)
                  Each keypoint: [x, y, z] in normalized coordinates
        """
        if self.hands_detector is None:
            return None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        # Process with MediaPipe
        results = self.hands_detector.process(image_rgb)

        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert normalized coordinates to pixel coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x * w,  # pixel x
                        'y': lm.y * h,  # pixel y
                        'z': lm.z,      # depth (relative)
                        'x_norm': lm.x, # normalized [0, 1]
                        'y_norm': lm.y  # normalized [0, 1]
                    })
                hands_data.append(landmarks)

        return {'hands': hands_data, 'image_size': (w, h)}

    def predict_sequential(self, image_path, robot_class='Research3'):
        """
        Run all models sequentially (single GPU or forced sequential).

        Args:
            image_path: Path to image file
            robot_class: Robot type

        Returns:
            dict with keys: 'robot', 'depth', 'human', 'timings'
        """
        timings = {}

        # Load image once
        image_pil = Image.open(image_path).convert('RGB')

        # 1. Robot Pose
        start = time.time()
        robot_result = self.predict_robot_pose(image_pil, robot_class)
        timings['robot'] = time.time() - start

        # 2. Depth
        start = time.time()
        depth_result = self.predict_depth(image_path)
        timings['depth'] = time.time() - start

        # 3. Human Pose
        start = time.time()
        human_result = self.predict_human_pose(image_path)
        timings['human'] = time.time() - start

        # 4. Hand Detection
        start = time.time()
        hands_result = self.predict_hands(image_path)
        timings['hands'] = time.time() - start

        timings['total'] = sum(timings.values())

        return {
            'robot': robot_result,
            'depth': depth_result,
            'human': human_result,
            'hands': hands_result,
            'timings': timings
        }

    def predict_parallel(self, image_path, robot_class='Research3'):
        """
        Run all models in parallel on different GPUs.

        Args:
            image_path: Path to image file
            robot_class: Robot type

        Returns:
            dict with keys: 'robot', 'depth', 'human', 'timings'
        """
        results = {}
        exceptions = {}
        timings = {}

        # Load image once
        image_pil = Image.open(image_path).convert('RGB')

        def robot_worker():
            try:
                start = time.time()
                results['robot'] = self.predict_robot_pose(image_pil, robot_class)
                timings['robot'] = time.time() - start
            except Exception as e:
                exceptions['robot'] = str(e)

        def depth_worker():
            try:
                start = time.time()
                results['depth'] = self.predict_depth(image_path)
                timings['depth'] = time.time() - start
            except Exception as e:
                exceptions['depth'] = str(e)

        def human_worker():
            try:
                start = time.time()
                results['human'] = self.predict_human_pose(image_path)
                timings['human'] = time.time() - start
            except Exception as e:
                exceptions['human'] = str(e)

        def hands_worker():
            try:
                start = time.time()
                results['hands'] = self.predict_hands(image_path)
                elapsed = time.time() - start
                timings['hands'] = elapsed
                # print(f"[DEBUG] Hands worker: {elapsed:.4f}s")  # Debug
            except Exception as e:
                exceptions['hands'] = str(e)

        # Start all threads
        threads = [
            threading.Thread(target=robot_worker, name='RobotThread'),
            threading.Thread(target=depth_worker, name='DepthThread'),
            threading.Thread(target=human_worker, name='HumanThread'),
            threading.Thread(target=hands_worker, name='HandsThread')
        ]

        start_total = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        timings['total'] = time.time() - start_total

        if exceptions:
            print(f"⚠ Errors occurred: {exceptions}")

        results['timings'] = timings
        return results

    def predict(self, image_path, robot_class='Research3'):
        """
        Run inference (automatically chooses parallel or sequential).

        Args:
            image_path: Path to image file
            robot_class: Robot type

        Returns:
            dict with keys: 'robot', 'depth', 'human', 'timings'
        """
        if self.use_multi_gpu:
            return self.predict_parallel(image_path, robot_class)
        else:
            return self.predict_sequential(image_path, robot_class)


def check_occlusion_by_depth_gradient(keypoints_2d, depth_map, w, h, gradient_threshold=0.3, window_size=7):
    """
    Check if a keypoint is likely occluded by examining depth gradient around it.

    If there's a sharp depth discontinuity (edge) near the keypoint, it's likely occluded.

    Args:
        keypoints_2d: Keypoints with confidence (N, 3) [x, y, conf]
        depth_map: Depth map (H, W)
        w, h: Image dimensions
        gradient_threshold: Depth gradient threshold (m/pixel) to consider as occlusion edge
        window_size: Window size to check around keypoint

    Returns:
        List of boolean (True = likely occluded)
    """
    occlusion_flags = []
    half_win = window_size // 2

    for kpt in keypoints_2d:
        x_2d, y_2d = int(kpt[0]), int(kpt[1])

        if not (0 <= x_2d < w and 0 <= y_2d < h):
            occlusion_flags.append(False)
            continue

        # Extract local window
        y_min = max(0, y_2d - half_win)
        y_max = min(h, y_2d + half_win + 1)
        x_min = max(0, x_2d - half_win)
        x_max = min(w, x_2d + half_win + 1)

        local_depth = depth_map[y_min:y_max, x_min:x_max]

        if local_depth.size == 0:
            occlusion_flags.append(False)
            continue

        center_depth = depth_map[y_2d, x_2d]

        # Check for large depth discontinuity (occlusion boundary)
        # If the keypoint is on an object that's occluding another part,
        # there will be a sharp depth change nearby
        depth_std = np.std(local_depth)
        depth_range = np.max(local_depth) - np.min(local_depth)

        # High variance and large range indicate occlusion boundary
        is_occluded = (depth_std > 0.2 and depth_range > 0.5)

        occlusion_flags.append(is_occluded)

    return occlusion_flags


def estimate_occluded_depth(keypoints_3d, skeleton_connections, keypoints_2d=None,
                           confidence_scores=None, occlusion_flags=None,
                           confidence_threshold=0.3, max_depth_diff=0.5):
    """
    Estimate depth for occluded keypoints using neighboring visible keypoints.

    Args:
        keypoints_3d: List of 3D points or None for each keypoint
        skeleton_connections: List of (idx1, idx2) tuples representing skeleton structure
        keypoints_2d: Optional 2D keypoints for additional checks
        confidence_scores: Optional confidence scores for each keypoint
        occlusion_flags: Optional list of boolean flags indicating occlusion
        confidence_threshold: Minimum confidence for a keypoint to be considered visible
        max_depth_diff: Maximum allowed depth difference (m) between connected joints

    Returns:
        Corrected list of 3D points
    """
    corrected_3d = keypoints_3d.copy()

    # Build adjacency graph
    adjacency = {}
    for idx1, idx2 in skeleton_connections:
        if idx1 not in adjacency:
            adjacency[idx1] = []
        if idx2 not in adjacency:
            adjacency[idx2] = []
        adjacency[idx1].append(idx2)
        adjacency[idx2].append(idx1)

    # Detect and correct outliers based on neighboring keypoints
    for i, pt in enumerate(corrected_3d):
        if pt is None:
            continue

        # Check if this point should be corrected
        should_correct = False

        # Method 1: Low confidence score
        if confidence_scores is not None and i < len(confidence_scores):
            if confidence_scores[i] < confidence_threshold:
                should_correct = True

        # Method 2: Depth gradient indicates occlusion
        if occlusion_flags is not None and i < len(occlusion_flags):
            if occlusion_flags[i]:
                should_correct = True

        # Method 3: Depth inconsistency with neighbors
        neighbors = adjacency.get(i, [])
        valid_neighbors = [corrected_3d[n] for n in neighbors if corrected_3d[n] is not None]

        if len(valid_neighbors) >= 2:
            # Calculate median depth of neighbors
            neighbor_depths = [n[2] for n in valid_neighbors]
            median_depth = np.median(neighbor_depths)

            # Calculate expected depth range based on neighbor variance
            depth_variance = np.std(neighbor_depths)
            adaptive_threshold = max(max_depth_diff, depth_variance * 2)

            # If current depth differs too much from median, likely occluded
            if abs(pt[2] - median_depth) > adaptive_threshold:
                should_correct = True

            # Apply correction if needed
            if should_correct:
                # Use X, Y from detection but Z (depth) from neighbor interpolation
                corrected_3d[i] = [pt[0], pt[1], median_depth]

    return corrected_3d


def smooth_skeleton_depth(keypoints_2d, depth_map, skeleton_connections, w, h, fx, fy, cx, cy,
                          confidence_threshold=0.3, max_bone_length=1.0, depth_smoothing_window=5):
    """
    Extract 3D keypoints with occlusion-aware depth correction.

    Args:
        keypoints_2d: YOLO keypoints with confidence (N, 3) where each row is [x, y, conf]
        depth_map: Depth map (H, W)
        skeleton_connections: List of (idx1, idx2) skeleton connections
        w, h: Image dimensions
        fx, fy, cx, cy: Camera intrinsics
        confidence_threshold: Minimum confidence for valid keypoint
        max_bone_length: Maximum expected bone length in meters
        depth_smoothing_window: Window size for local depth smoothing

    Returns:
        List of 3D points (may contain None for low-confidence keypoints)
    """
    # Step 1: Check for occlusion using depth gradient analysis
    occlusion_flags = check_occlusion_by_depth_gradient(
        keypoints_2d, depth_map, w, h,
        gradient_threshold=0.3,
        window_size=7
    )

    # Step 2: Extract confidence scores
    confidence_scores = [kpt[2] for kpt in keypoints_2d]

    # Step 3: Convert 2D keypoints to 3D using depth
    keypoints_3d = []

    for kpt in keypoints_2d:
        if kpt[2] > confidence_threshold:
            x_2d, y_2d = int(kpt[0]), int(kpt[1])

            if 0 <= x_2d < w and 0 <= y_2d < h:
                # Instead of using single pixel depth, use local median for robustness
                half_win = depth_smoothing_window // 2
                y_min = max(0, y_2d - half_win)
                y_max = min(h, y_2d + half_win + 1)
                x_min = max(0, x_2d - half_win)
                x_max = min(w, x_2d + half_win + 1)

                local_depths = depth_map[y_min:y_max, x_min:x_max]
                # Use median instead of mean to be robust to outliers
                Z = np.median(local_depths)

                X = (x_2d - cx) * Z / fx
                Y = (y_2d - cy) * Z / fy
                keypoints_3d.append([X, Y, Z])
            else:
                keypoints_3d.append(None)
        else:
            keypoints_3d.append(None)

    # Step 4: Apply multi-method occlusion-aware depth correction
    keypoints_3d = estimate_occluded_depth(
        keypoints_3d, skeleton_connections,
        keypoints_2d=keypoints_2d,
        confidence_scores=confidence_scores,
        occlusion_flags=occlusion_flags,
        confidence_threshold=confidence_threshold,
        max_depth_diff=0.5
    )

    return keypoints_3d


def calculate_minimum_distance(robot_3d, human_3d_list):
    """
    Calculate minimum distance between robot and all humans.

    Args:
        robot_3d: (N, 3) array of robot 3D points
        human_3d_list: List of human 3D points arrays

    Returns:
        dict with:
            - 'min_distance': minimum distance in meters
            - 'robot_point_idx': index of closest robot point
            - 'human_idx': index of closest human
            - 'human_point_idx': index of closest human point
            - 'robot_point': 3D coordinates of robot point
            - 'human_point': 3D coordinates of human point
    """
    if robot_3d is None or len(human_3d_list) == 0:
        return None

    min_dist = float('inf')
    closest_info = None

    for human_idx, human_3d in enumerate(human_3d_list):
        if human_3d is None or len(human_3d) == 0:
            continue

        # Calculate all pairwise distances
        for i, r_pt in enumerate(robot_3d):
            for j, h_pt in enumerate(human_3d):
                if h_pt is not None:
                    dist = np.linalg.norm(r_pt - h_pt)
                    if dist < min_dist:
                        min_dist = dist
                        closest_info = {
                            'min_distance': dist,
                            'robot_point_idx': i,
                            'human_idx': human_idx,
                            'human_point_idx': j,
                            'robot_point': r_pt.copy(),
                            'human_point': h_pt.copy()
                        }

    return closest_info


def get_danger_level(distance):
    """
    Determine danger level based on distance.

    Args:
        distance: Distance in meters

    Returns:
        tuple of (level_name, color_rgb)
    """
    if distance < 0.1:  # < 30cm
        return "CRITICAL", (255, 0, 0)  # Red
    elif distance < 0.3:  # < 50cm
        return "WARNING", (255, 165, 0)  # Orange
    elif distance < 0.5:  # < 80cm
        return "CAUTION", (255, 255, 0)  # Yellow
    else:
        return "SAFE", (0, 255, 0)  # Green


def project_3d_to_2d(points_3d, w, h):
    """
    Project 3D points in camera coordinates back to 2D image plane.

    Args:
        points_3d: (N, 3) array of 3D points in camera coordinates [X, Y, Z]
        w, h: Image dimensions

    Returns:
        (N, 2) array of 2D image coordinates [u, v]
    """
    # ZED camera intrinsic parameters (at 1920x1080 resolution)
    fx_original, fy_original = 1072.56, 1073.69
    cx_original, cy_original = 978.568, 557.972
    w_original, h_original = 1920, 1080

    # Scale intrinsics to current image resolution
    fx = fx_original * (w / w_original)
    fy = fy_original * (h / h_original)
    cx = cx_original * (w / w_original)
    cy = cy_original * (h / h_original)

    # Project to 2D
    points_2d = []
    for pt_3d in points_3d:
        if pt_3d is None:
            points_2d.append(None)
        else:
            X, Y, Z = pt_3d
            if Z > 0:  # Only project points in front of camera
                u = (X * fx / Z) + cx
                v = (Y * fy / Z) + cy
                points_2d.append([u, v])
            else:
                points_2d.append(None)

    return points_2d


def apply_rotation_offset(points_3d, rot_z_deg=180, rot_x_deg=90):
    """
    Apply rotation offset to 3D points.

    Args:
        points_3d: (N, 3) array of 3D points
        rot_z_deg: Rotation around Z-axis in degrees
        rot_x_deg: Rotation around X-axis in degrees

    Returns:
        Rotated 3D points
    """
    # Convert degrees to radians
    rot_z_rad = np.radians(rot_z_deg)
    rot_x_rad = np.radians(rot_x_deg)

    # Rotation matrix around Z-axis
    Rz = np.array([
        [np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
        [np.sin(rot_z_rad),  np.cos(rot_z_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
        [0, np.sin(rot_x_rad),  np.cos(rot_x_rad)]
    ])

    # Combined rotation: first Z, then X
    R = Rx @ Rz

    # Apply rotation
    rotated_points = points_3d @ R.T

    return rotated_points


def draw_3d_scene(ax, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View", view_elev=20, view_azim=45, view_roll=45):
    """
    Draw complete 3D scene with robot, human, and hands.

    Args:
        ax: Matplotlib 3D axis
        results: Prediction results
        robot_keypoints: Robot 2D keypoints
        human_keypoints: Human 2D keypoints
        human_skeleton: Human skeleton connections
        w, h: Image dimensions
        title: Title for the subplot
        view_elev: Elevation angle for 3D view
        view_azim: Azimuth angle for 3D view
    """
    robot_3d_depth = None

    # Draw robot 3D structure - Depth-based only
    if results.get('robot') and results.get('depth') is not None and robot_keypoints is not None:
        depth_resized = cv2.resize(results['depth'], (w, h))

        # ZED camera intrinsics
        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080

        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        robot_3d_depth = []
        for kpt_2d in robot_keypoints:
            x_2d, y_2d = int(kpt_2d[0]), int(kpt_2d[1])
            if 0 <= x_2d < w and 0 <= y_2d < h:
                Z = depth_resized[y_2d, x_2d]
                X = (x_2d - cx) * Z / fx
                Y = (y_2d - cy) * Z / fy
                robot_3d_depth.append([X, Y, Z])

        if len(robot_3d_depth) > 0:
            robot_3d_depth = np.array(robot_3d_depth)
            ax.scatter(robot_3d_depth[:, 0], robot_3d_depth[:, 1], robot_3d_depth[:, 2],
                      c='blue', s=80, marker='o', label='Robot',
                      edgecolors='darkblue', linewidths=2)

            # Add joint numbers
            for i, pt in enumerate(robot_3d_depth):
                ax.text(pt[0], pt[1], pt[2], f'{i}', fontsize=9, color='black',
                        weight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='yellow',
                                 edgecolor='black', linewidth=1, alpha=0.9))

            # Draw connections with cylindrical mesh (thinner)
            for i in range(len(robot_3d_depth) - 1):
                cylinder = create_cylinder_mesh(robot_3d_depth[i], robot_3d_depth[i+1],
                                              radius=0.015, num_segments=8)  # radius: 50mm -> 15mm
                if cylinder is not None:
                    cylinder.set_facecolor('blue')
                    cylinder.set_alpha(0.4)  # More transparent
                    ax.add_collection3d(cylinder)

    # Draw human 3D pose with occlusion correction
    if results.get('depth') is not None and len(human_keypoints) > 0:
        depth_resized = cv2.resize(results['depth'], (w, h))

        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080

        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        for person_idx, person_kpts in enumerate(human_keypoints):
            human_3d_full = smooth_skeleton_depth(
                person_kpts, depth_resized, human_skeleton,
                w, h, fx, fy, cx, cy,
                confidence_threshold=0.3, max_bone_length=1.0, depth_smoothing_window=5
            )

            human_3d_points = [pt for pt in human_3d_full if pt is not None]
            if human_3d_points:
                human_3d_points = np.array(human_3d_points)
                ax.scatter(human_3d_points[:, 0], human_3d_points[:, 1], human_3d_points[:, 2],
                          c='green', s=50, marker='^', alpha=0.7,
                          label=f'Human {person_idx+1}' if person_idx == 0 else '')

            # Draw skeleton
            for idx1, idx2 in human_skeleton:
                if human_3d_full[idx1] is not None and human_3d_full[idx2] is not None:
                    pt1, pt2 = human_3d_full[idx1], human_3d_full[idx2]
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
                           'g-', linewidth=2, alpha=0.6)

    # Draw hands
    if results.get('depth') is not None and results.get('hands') and results['hands'] is not None:
        depth_resized = cv2.resize(results['depth'], (w, h))
        hands_data = results['hands']['hands']

        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080

        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        for hand_idx, hand_landmarks in enumerate(hands_data):
            hand_kpts_2d = np.array([[lm['x'], lm['y'], 1.0] for lm in hand_landmarks])
            hand_3d_points = smooth_skeleton_depth(
                hand_kpts_2d, depth_resized, hand_connections,
                w, h, fx, fy, cx, cy,
                confidence_threshold=0.5, max_bone_length=0.3, depth_smoothing_window=3
            )

            valid_hand_3d = [pt for pt in hand_3d_points if pt is not None]
            if valid_hand_3d:
                valid_hand_3d = np.array(valid_hand_3d)
                ax.scatter(valid_hand_3d[:, 0], valid_hand_3d[:, 1], valid_hand_3d[:, 2],
                          c='cyan', s=30, marker='*', alpha=0.8,
                          label=f'Hand {hand_idx+1}' if hand_idx == 0 else '')

            # Draw connections
            for idx1, idx2 in hand_connections:
                if (idx1 < len(hand_3d_points) and idx2 < len(hand_3d_points) and
                    hand_3d_points[idx1] is not None and hand_3d_points[idx2] is not None):
                    pt1, pt2 = hand_3d_points[idx1], hand_3d_points[idx2]
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
                           'c-', linewidth=1.5, alpha=0.7)

    # Calculate and visualize minimum distance between robot and humans
    if robot_3d_depth is not None and results.get('depth') is not None and len(human_keypoints) > 0:
        # Collect all human 3D points
        depth_resized = cv2.resize(results['depth'], (w, h))
        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080
        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        human_3d_list = []
        for person_kpts in human_keypoints:
            human_3d_full = smooth_skeleton_depth(
                person_kpts, depth_resized, human_skeleton,
                w, h, fx, fy, cx, cy,
                confidence_threshold=0.3, max_bone_length=1.0, depth_smoothing_window=5
            )
            human_3d_valid = [pt for pt in human_3d_full if pt is not None]
            if human_3d_valid:
                human_3d_list.append(np.array(human_3d_valid))

        # Calculate minimum distance
        closest = calculate_minimum_distance(robot_3d_depth, human_3d_list)

        if closest:
            min_dist = closest['min_distance']
            danger_level, danger_color = get_danger_level(min_dist)

            # Draw line between closest points
            r_pt = closest['robot_point']
            h_pt = closest['human_point']

            # Convert RGB to matplotlib format (0-1 range)
            line_color = tuple(c / 255.0 for c in danger_color)

            ax.plot([r_pt[0], h_pt[0]], [r_pt[1], h_pt[1]], [r_pt[2], h_pt[2]],
                   color=line_color, linewidth=3, linestyle='--', alpha=0.8)

            # Mark closest points
            ax.scatter([r_pt[0]], [r_pt[1]], [r_pt[2]],
                      c=[line_color], s=150, marker='*', edgecolors='black', linewidths=2)
            ax.scatter([h_pt[0]], [h_pt[1]], [h_pt[2]],
                      c=[line_color], s=150, marker='*', edgecolors='black', linewidths=2)

            # Add distance text at midpoint
            mid_pt = (r_pt + h_pt) / 2
            ax.text(mid_pt[0], mid_pt[1], mid_pt[2],
                   f'{min_dist:.2f}m\n{danger_level}',
                   fontsize=10, color='black', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=line_color,
                            edgecolor='black', linewidth=2, alpha=0.9))

    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)


def create_cylinder_mesh(start_point, end_point, radius=0.05, num_segments=10):
    """
    Create a cylindrical mesh between two 3D points.

    Args:
        start_point: (x, y, z) starting point
        end_point: (x, y, z) ending point
        radius: cylinder radius in meters (default: 0.05m = 50mm)
        num_segments: number of segments for cylinder smoothness

    Returns:
        Poly3DCollection representing the cylinder
    """
    start = np.array(start_point)
    end = np.array(end_point)

    # Direction vector
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None

    # Normalized direction
    v = vec / length

    # Find perpendicular vectors
    not_v = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, num_segments)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    # Create cylinder vertices
    vertices = []
    for i in range(num_segments):
        # Bottom circle
        point_bottom = start + circle_x[i] * n1 + circle_y[i] * n2
        vertices.append(point_bottom)
        # Top circle
        point_top = end + circle_x[i] * n1 + circle_y[i] * n2
        vertices.append(point_top)

    # Create faces
    faces = []
    for i in range(num_segments):
        # Side face (quad made of two triangles)
        idx1 = 2 * i
        idx2 = 2 * i + 1
        idx3 = 2 * ((i + 1) % num_segments) + 1
        idx4 = 2 * ((i + 1) % num_segments)

        faces.append([vertices[idx1], vertices[idx2], vertices[idx3], vertices[idx4]])

    return Poly3DCollection(faces, alpha=0.6, linewidths=0.5, edgecolors='none')


def visualize_results(image_path, results, output_path):
    """
    Visualize all prediction results in a single figure with integrated views.

    Args:
        image_path: Path to original image
        results: Prediction results dict
        output_path: Path to save visualization
    """
    # Load original image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # COCO skeleton connections for human pose
    human_skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    # Extract human keypoints
    human_keypoints = []
    num_persons = 0
    if results.get('human') and results['human'] is not None:
        yolo_result = results['human']
        if hasattr(yolo_result, 'keypoints') and yolo_result.keypoints is not None:
            keypoints_data = yolo_result.keypoints
            if hasattr(keypoints_data, 'data'):
                human_keypoints = keypoints_data.data.cpu().numpy()
                num_persons = len(human_keypoints)

    # Extract robot keypoints
    robot_keypoints = None
    if results.get('robot'):
        robot_keypoints = results['robot']['keypoints_2d']

    fig = plt.figure(figsize=(24, 18))

    # 1. INTEGRATED: Robot + Human Pose on Original Image
    ax1 = plt.subplot(3, 4, 1)
    img_integrated = img_rgb.copy()

    # Draw robot skeleton (green)
    if robot_keypoints is not None:
        for i in range(len(robot_keypoints) - 1):
            pt1 = tuple(robot_keypoints[i].astype(int))
            pt2 = tuple(robot_keypoints[i + 1].astype(int))
            cv2.line(img_integrated, pt1, pt2, (0, 255, 0), 3)
        for i, pt in enumerate(robot_keypoints):
            pt_int = tuple(pt.astype(int))
            cv2.circle(img_integrated, pt_int, 6, (255, 0, 0), -1)
            cv2.putText(img_integrated, f'R{i}', (pt_int[0]+5, pt_int[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw human skeleton (blue/red)
    for person_kpts in human_keypoints:
        for idx1, idx2 in human_skeleton:
            if person_kpts[idx1][2] > 0.3 and person_kpts[idx2][2] > 0.3:
                pt1 = (int(person_kpts[idx1][0]), int(person_kpts[idx1][1]))
                pt2 = (int(person_kpts[idx2][0]), int(person_kpts[idx2][1]))
                cv2.line(img_integrated, pt1, pt2, (255, 0, 255), 2)
        for kpt in person_kpts:
            if kpt[2] > 0.3:
                pt = (int(kpt[0]), int(kpt[1]))
                cv2.circle(img_integrated, pt, 4, (255, 255, 0), -1)

    # Draw hand keypoints (MediaPipe Hands - 21 landmarks per hand)
    num_hands = 0
    if results.get('hands') and results['hands'] is not None:
        hands_data = results['hands']['hands']
        num_hands = len(hands_data)

        # MediaPipe hand connections (fingers)
        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        for hand_idx, hand_landmarks in enumerate(hands_data):
            # Draw connections
            for idx1, idx2 in hand_connections:
                pt1 = (int(hand_landmarks[idx1]['x']), int(hand_landmarks[idx1]['y']))
                pt2 = (int(hand_landmarks[idx2]['x']), int(hand_landmarks[idx2]['y']))
                cv2.line(img_integrated, pt1, pt2, (0, 255, 255), 2)  # Cyan for hands

            # Draw keypoints
            for lm_idx, lm in enumerate(hand_landmarks):
                pt = (int(lm['x']), int(lm['y']))
                if lm_idx in [0, 4, 8, 12, 16, 20]:  # Fingertips and wrist
                    cv2.circle(img_integrated, pt, 5, (255, 0, 0), -1)  # Red for fingertips
                else:
                    cv2.circle(img_integrated, pt, 3, (0, 255, 255), -1)  # Cyan for joints

    ax1.imshow(img_integrated)
    title_text = f'INTEGRATED: Robot (Green) + Human (Magenta) [{num_persons} persons]'
    if num_hands > 0:
        title_text += f' + Hands (Cyan) [{num_hands} hands]'
    ax1.set_title(title_text, fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. INTEGRATED: Depth Map with Keypoints Overlay
    ax2 = plt.subplot(3, 4, 2)
    if results.get('depth') is not None:
        depth_resized = cv2.resize(results['depth'], (w, h))

        # Normalize depth for visualization
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min())
        depth_colored = (plt.cm.turbo(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

        # Overlay robot keypoints (green)
        if robot_keypoints is not None:
            for i in range(len(robot_keypoints) - 1):
                pt1 = tuple(robot_keypoints[i].astype(int))
                pt2 = tuple(robot_keypoints[i + 1].astype(int))
                cv2.line(depth_colored, pt1, pt2, (0, 255, 0), 2)
            for pt in robot_keypoints:
                pt_int = tuple(pt.astype(int))
                cv2.circle(depth_colored, pt_int, 5, (255, 0, 0), -1)

        # Overlay human keypoints (yellow)
        for person_kpts in human_keypoints:
            for idx1, idx2 in human_skeleton:
                if person_kpts[idx1][2] > 0.3 and person_kpts[idx2][2] > 0.3:
                    pt1 = (int(person_kpts[idx1][0]), int(person_kpts[idx1][1]))
                    pt2 = (int(person_kpts[idx2][0]), int(person_kpts[idx2][1]))
                    cv2.line(depth_colored, pt1, pt2, (255, 255, 0), 2)
            for kpt in person_kpts:
                if kpt[2] > 0.3:
                    pt = (int(kpt[0]), int(kpt[1]))
                    cv2.circle(depth_colored, pt, 4, (255, 255, 255), -1)

        # Overlay hand keypoints (cyan)
        if results.get('hands') and results['hands'] is not None:
            hands_data = results['hands']['hands']
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for hand_landmarks in hands_data:
                for idx1, idx2 in hand_connections:
                    pt1 = (int(hand_landmarks[idx1]['x']), int(hand_landmarks[idx1]['y']))
                    pt2 = (int(hand_landmarks[idx2]['x']), int(hand_landmarks[idx2]['y']))
                    cv2.line(depth_colored, pt1, pt2, (0, 255, 255), 2)
                for lm in hand_landmarks:
                    pt = (int(lm['x']), int(lm['y']))
                    cv2.circle(depth_colored, pt, 3, (255, 255, 255), -1)

        ax2.imshow(depth_colored)
        ax2.set_title('INTEGRATED: Depth + Keypoints (w/ Hands)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Depth Not Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Depth Map with Keypoints', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Robot Pose Only
    ax3 = plt.subplot(3, 4, 3)
    img_robot = img_rgb.copy()
    if robot_keypoints is not None:
        for i in range(len(robot_keypoints) - 1):
            pt1 = tuple(robot_keypoints[i].astype(int))
            pt2 = tuple(robot_keypoints[i + 1].astype(int))
            cv2.line(img_robot, pt1, pt2, (0, 255, 0), 3)
        for i, pt in enumerate(robot_keypoints):
            pt_int = tuple(pt.astype(int))
            cv2.circle(img_robot, pt_int, 5, (255, 0, 0), -1)
    ax3.imshow(img_robot)
    ax3.set_title('Robot Pose Only', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Human Pose Only
    ax4 = plt.subplot(3, 4, 4)
    img_human = img_rgb.copy()
    for person_kpts in human_keypoints:
        for idx1, idx2 in human_skeleton:
            if person_kpts[idx1][2] > 0.3 and person_kpts[idx2][2] > 0.3:
                pt1 = (int(person_kpts[idx1][0]), int(person_kpts[idx1][1]))
                pt2 = (int(person_kpts[idx2][0]), int(person_kpts[idx2][1]))
                cv2.line(img_human, pt1, pt2, (255, 0, 0), 2)
        for kpt in person_kpts:
            if kpt[2] > 0.3:
                pt = (int(kpt[0]), int(kpt[1]))
                cv2.circle(img_human, pt, 3, (0, 255, 0), -1)
    ax4.imshow(img_human)
    ax4.set_title(f'Human Pose Only ({num_persons} persons)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5-10. Multiple 3D Views from different angles
    # View 1: Front-Right (elev=20, azim=45)
    ax5 = plt.subplot(3, 4, 5, projection='3d')
    draw_3d_scene(ax5, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Front-Right", view_elev=100, view_azim=90, view_roll=0)

    # View 2: Front-Left (elev=20, azim=135)
    ax6 = plt.subplot(3, 4, 6, projection='3d')
    draw_3d_scene(ax6, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Front-Left (135°)", view_elev=120, view_azim=90, view_roll=0)

    # View 3: Top view (elev=80, azim=45)
    ax7 = plt.subplot(3, 4, 7, projection='3d')
    draw_3d_scene(ax7, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Top (80°)", view_elev=140, view_azim=90, view_roll=0)

    # View 4: Side view (elev=0, azim=90)
    ax8 = plt.subplot(3, 4, 8, projection='3d')
    draw_3d_scene(ax8, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Side (90°)", view_elev=160, view_azim=90, view_roll=0)

    # View 5: Back view (elev=10, azim=225)
    ax9 = plt.subplot(3, 4, 9, projection='3d')
    draw_3d_scene(ax9, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Back (225°)", view_elev=170, view_azim=90, view_roll=0)

    # View 6: Isometric (elev=30, azim=300)
    ax10 = plt.subplot(3, 4, 10, projection='3d')
    draw_3d_scene(ax10, results, robot_keypoints, human_keypoints, human_skeleton, w, h,
                  title="3D View: Isometric (300°)", view_elev=175, view_azim=90, view_roll=0)

    # 11. Timing Info + Safety Distance
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    # Calculate minimum distance for info display
    safety_text = ""
    if results.get('robot') and results.get('depth') is not None and robot_keypoints is not None and len(human_keypoints) > 0:
        depth_resized = cv2.resize(results['depth'], (w, h))
        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080
        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        # Get robot 3D
        robot_3d_depth = []
        for kpt_2d in robot_keypoints:
            x_2d, y_2d = int(kpt_2d[0]), int(kpt_2d[1])
            if 0 <= x_2d < w and 0 <= y_2d < h:
                Z = depth_resized[y_2d, x_2d]
                X = (x_2d - cx) * Z / fx
                Y = (y_2d - cy) * Z / fy
                robot_3d_depth.append([X, Y, Z])
        if len(robot_3d_depth) > 0:
            robot_3d_depth = np.array(robot_3d_depth)

            # Get human 3D
            human_3d_list = []
            for person_kpts in human_keypoints:
                human_3d_full = smooth_skeleton_depth(
                    person_kpts, depth_resized, human_skeleton,
                    w, h, fx, fy, cx, cy,
                    confidence_threshold=0.3, max_bone_length=1.0, depth_smoothing_window=5
                )
                human_3d_valid = [pt for pt in human_3d_full if pt is not None]
                if human_3d_valid:
                    human_3d_list.append(np.array(human_3d_valid))

            # Calculate minimum distance
            closest = calculate_minimum_distance(robot_3d_depth, human_3d_list)
            if closest:
                min_dist = closest['min_distance']
                danger_level, _ = get_danger_level(min_dist)
                safety_text = (
                    f"\n{'═' * 32}\n"
                    f"⚠️  Safety Distance\n"
                    f"  • Min Distance: {min_dist:.3f}m\n"
                    f"  • Status: {danger_level}\n"
                    f"  • Robot Joint: {closest['robot_point_idx']}\n"
                    f"  • Human #{closest['human_idx']+1} Joint: {closest['human_point_idx']}\n"
                )

    if results.get('timings'):
        timings = results['timings']
        timing_text = (
            f"⏱ Inference Timings\n"
            f"{'═' * 32}\n\n"
            f"Robot Pose:  {timings.get('robot', 0):.3f}s\n"
            f"Depth Map:   {timings.get('depth', 0):.3f}s\n"
            f"Human Pose:  {timings.get('human', 0):.3f}s\n"
            f"Hand Detect: {timings.get('hands', 0):.3f}s\n\n"
            f"{'─' * 32}\n"
            f"Total:       {timings.get('total', 0):.3f}s\n"
            f"FPS:         {1.0/timings.get('total', 1):.2f}\n\n"
            f"{'═' * 32}\n"
            f"Detections:\n"
            f"  • Robot: {'✓' if results.get('robot') else '✗'}\n"
            f"  • Depth: {'✓' if results.get('depth') is not None else '✗'}\n"
            f"  • Humans: {num_persons}\n"
            f"  • Hands: {num_hands}\n\n"
            f"{'═' * 32}\n"
            f"3D Reconstruction:\n"
            f"  • Method: Depth-based\n"
            f"  • Camera: ZED Stereo"
            f"{safety_text}"
        )
        ax11.text(0.05, 0.5, timing_text, fontsize=9, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

    # 12. 3D to 2D Reprojection Verification
    ax12 = plt.subplot(3, 4, 12)
    img_reprojection = img_rgb.copy()

    # Reproject depth-based 3D robot back to 2D
    if results.get('robot') and results.get('depth') is not None and robot_keypoints is not None:
        depth_resized = cv2.resize(results['depth'], (w, h))
        fx_original, fy_original = 1072.56, 1073.69
        cx_original, cy_original = 978.568, 557.972
        w_original, h_original = 1920, 1080
        fx = fx_original * (w / w_original)
        fy = fy_original * (h / h_original)
        cx = cx_original * (w / w_original)
        cy = cy_original * (h / h_original)

        # Get depth-based 3D points
        robot_3d_depth = []
        for kpt_2d in robot_keypoints:
            x_2d, y_2d = int(kpt_2d[0]), int(kpt_2d[1])
            if 0 <= x_2d < w and 0 <= y_2d < h:
                Z = depth_resized[y_2d, x_2d]
                X = (x_2d - cx) * Z / fx
                Y = (y_2d - cy) * Z / fy
                robot_3d_depth.append([X, Y, Z])
            else:
                robot_3d_depth.append(None)

        # Reproject to 2D
        robot_2d_reprojected = project_3d_to_2d(robot_3d_depth, w, h)

        # Draw original 2D keypoints (green)
        for i, pt in enumerate(robot_keypoints):
            pt_int = tuple(pt.astype(int))
            cv2.circle(img_reprojection, pt_int, 5, (0, 255, 0), -1)  # Green
            cv2.putText(img_reprojection, f'{i}', (pt_int[0]+5, pt_int[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw reprojected 2D keypoints (magenta)
        for i, pt_2d in enumerate(robot_2d_reprojected):
            if pt_2d is not None:
                pt_int = (int(pt_2d[0]), int(pt_2d[1]))
                if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
                    cv2.circle(img_reprojection, pt_int, 5, (255, 0, 255), 2)  # Magenta circle
                    cv2.line(img_reprojection,
                            tuple(robot_keypoints[i].astype(int)),
                            pt_int,
                            (255, 255, 0), 1)  # Yellow line showing reprojection error

    ax12.imshow(img_reprojection)
    ax12.set_title('3D→2D Reprojection\n(Green: Original, Magenta: Reprojected)', fontsize=10, fontweight='bold')
    ax12.axis('off')

    plt.suptitle('Integrated Multi-Model Pipeline: Robot + Human Pose & Depth Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Integrated visualization saved to {output_path}")


def main(args):
    # Initialize pipeline
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

    # Run inference
    print(f"\nRunning inference on {args.image_path}...")
    results = pipeline.predict(args.image_path, robot_class=args.robot_class)

    # Print timings
    print("\n" + "=" * 80)
    print("Inference Results:")
    print("=" * 80)
    if results.get('timings'):
        timings = results['timings']
        print(f"Robot Pose:  {timings.get('robot', 0):.3f}s")
        print(f"Depth:       {timings.get('depth', 0):.3f}s")
        print(f"Human Pose:  {timings.get('human', 0):.3f}s")
        print(f"{'─' * 80}")
        print(f"Total:       {timings.get('total', 0):.3f}s")
    print("=" * 80)

    # Visualize
    if args.output_path:
        visualize_results(args.image_path, results, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Multi-Model Pipeline")

    # Robot model
    parser.add_argument("--robot_checkpoint", type=str, required=True, help="Robot pose model checkpoint")
    parser.add_argument("--robot_model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--robot_heatmap_size", type=str, default="640,360", help="H,W")
    parser.add_argument("--robot_class", type=str, default="Research3", help="Robot type")

    # Depth model
    parser.add_argument("--depth_model_name", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")

    # YOLO-Pose model
    parser.add_argument("--yolo_pose_model", type=str, default="yolov8l-pose.pt", help="YOLO pose model (e.g., yolov8l-pose.pt)")

    # GPU settings
    parser.add_argument("--use_multi_gpu", action="store_true", help="Use multi-GPU for parallel execution")
    parser.add_argument("--robot_gpu", type=int, default=0)
    parser.add_argument("--depth_gpu", type=int, default=1)
    parser.add_argument("--human_gpu", type=int, default=2)

    # Input/Output
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--output_path", type=str, default="integrated_result.png", help="Output visualization path")

    args = parser.parse_args()
    main(args)
