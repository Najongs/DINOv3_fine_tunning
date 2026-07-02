# Integrated Pipeline Setup Guide

통합 파이프라인 (Robot Pose + Depth + Human Pose) 설정 가이드입니다.

## 1. 필요한 라이브러리 설치

### MMPose & MMDetection 설치

```bash
# OpenMMLab 도구 설치
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

### Depth Anything 3 설치

```bash
# 이미 설치되어 있다면 skip
pip install depth-anything-v2
```

## 2. RTMPose 모델 다운로드

### 방법 1: 자동 다운로드 (추천)

파이프라인이 처음 실행될 때 자동으로 다운로드됩니다. 하지만 수동으로 미리 받아두는 것을 권장합니다.

### 방법 2: 수동 다운로드

```bash
cd /home/najo/NAS/DIP/DINOv3_fine_tunning

# Config 디렉토리 생성
mkdir -p configs
mkdir -p checkpoints

# RTMDet (Person Detection) Config 다운로드
wget -P configs/ https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py

# RTMDet Checkpoint 다운로드
wget -P checkpoints/ https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

# RTMPose-l Config 다운로드
wget -P configs/ https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py

# RTMPose-l Checkpoint 다운로드
wget -P checkpoints/ https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth
```

**더 가벼운 RTMPose-m 사용하고 싶다면:**

```bash
# RTMPose-m Config
wget -P configs/ https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py

# RTMPose-m Checkpoint
wget -P checkpoints/ https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

## 3. 파이프라인 설정

`run_integrated_pipeline.sh` 파일을 수정:

```bash
# Robot Pose Model
ROBOT_CHECKPOINT="checkpoints_pose/best_model.pth"  # 실제 경로로 수정
ROBOT_CLASS="Research3"  # 로봇 타입 설정

# RTMPose Model
RTM_DET_CONFIG="configs/rtmdet_m_640-8xb32_coco-person.py"
RTM_DET_CHECKPOINT="checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
RTM_POSE_CONFIG="configs/rtmpose-l_8xb256-420e_coco-256x192.py"
RTM_POSE_CHECKPOINT="checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth"

# Input/Output
IMAGE_PATH="test_image.jpg"  # 테스트 이미지 경로
OUTPUT_PATH="integrated_result.png"
```

## 4. 사용 방법

### 단일 이미지 테스트

```bash
# 스크립트 실행 (Multi-GPU 병렬)
./run_integrated_pipeline.sh
```

### Python에서 직접 사용

```python
from integrated_pipeline import IntegratedPipeline

# Initialize pipeline
pipeline = IntegratedPipeline(
    robot_checkpoint="checkpoints_pose/best_model.pth",
    robot_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    robot_heatmap_size=(640, 360),
    rtm_det_config="configs/rtmdet_m_640-8xb32_coco-person.py",
    rtm_det_checkpoint="checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    rtm_pose_config="configs/rtmpose-l_8xb256-420e_coco-256x192.py",
    rtm_pose_checkpoint="checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
    use_multi_gpu=True,
    robot_gpu=0,
    depth_gpu=1,
    human_gpu=2
)

# Run inference
results = pipeline.predict("test_image.jpg", robot_class="Research3")

# Access results
robot_pose = results['robot']  # {'heatmaps', 'angles', 'keypoints_2d', '3d_points'}
depth_map = results['depth']   # (H, W) numpy array
human_poses = results['human']  # List of pose results
timings = results['timings']    # Execution times
```

## 5. GPU 설정

### Multi-GPU (병렬 실행) - GPU 3개 이상

```bash
USE_MULTI_GPU="--use_multi_gpu"
ROBOT_GPU=0
DEPTH_GPU=1
HUMAN_GPU=2
```

**예상 시간:** ~0.2초 (가장 느린 모델 기준)

### Single GPU (순차 실행) - GPU 1개

```bash
USE_MULTI_GPU=""  # 비활성화
ROBOT_GPU=0
DEPTH_GPU=0
HUMAN_GPU=0
```

**예상 시간:** ~0.35초 (모든 모델 시간 합)

## 6. 출력 결과

통합 시각화 이미지 포함:
1. **Robot Pose** - 로봇 관절 skeleton 오버레이
2. **Depth Map** - 깊이 맵 (turbo colormap)
3. **Human Pose** - 사람 skeleton (여러 명 지원)
4. **Robot Heatmaps** - 로봇 관절 heatmap
5. **Robot 3D Structure** - 3D 관절 위치
6. **Timing Info** - 각 모델별 실행 시간

## 7. 모델별 예상 시간

| 모델 | Single GPU | Multi-GPU |
|------|-----------|-----------|
| Robot Pose (DINOv3) | 0.05초 | 0.05초 |
| Depth (DA3 GIANT) | 0.20초 | 0.20초 |
| Human Pose (RTMPose-l) | 0.11초 | 0.11초 |
| **Total** | **0.36초** | **0.20초** |

## 8. Troubleshooting

### MMPose 설치 오류

```bash
# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 버전과 맞는 mmcv 설치
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### Depth Anything 메모리 부족

```bash
# 더 작은 모델 사용
DEPTH_MODEL_NAME="depth-anything/DA3NESTED-LARGE"  # GIANT 대신 LARGE
```

### RTMPose Config 로드 오류

Config 파일들이 절대 경로로 설정되어 있는지 확인:

```bash
# 절대 경로 사용
RTM_DET_CONFIG="/home/najo/NAS/DIP/DINOv3_fine_tunning/configs/rtmdet_m_640-8xb32_coco-person.py"
```

## 9. RTMPose 모델 선택 가이드

| Model | Input Size | COCO AP | Speed | 메모리 | 추천 용도 |
|-------|-----------|---------|-------|--------|----------|
| RTMPose-t | 256x192 | 68.5 | ~350 FPS | ~1GB | 엣지/모바일 |
| RTMPose-s | 256x192 | 71.1 | ~200 FPS | ~1.5GB | 실시간 |
| RTMPose-m | 256x192 | 75.8 | ~150 FPS | ~2GB | **균형잡힌 선택** ✅ |
| RTMPose-l | 256x192 | 76.5 | ~90 FPS | ~3GB | 고정확도 |
| RTMPose-x | 256x192 | 77.8 | ~70 FPS | ~4GB | 최고 정확도 |

**추천:** Multi-GPU면 RTMPose-l, Single GPU면 RTMPose-m
