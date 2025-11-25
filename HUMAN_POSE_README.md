# Human Pose Estimation with DINOv3

이 문서는 DINOv3를 백본으로 사용하는 human pose estimation 모델의 학습 가이드입니다.

## 개요

- **목적**: 사람의 2D keypoint 자세를 추정 (COCO 17 keypoints)
- **백본**: Frozen DINOv3 (기존 로봇 자세 추정과 동일한 feature 사용)
- **출력**: 17개 keypoint에 대한 heatmap (Gaussian)
- **향후 계획**: 학습 완료 후 로봇 자세 추정 모델과 통합

## 파일 구조

```
├── human_pose_model.py          # Human pose estimation 모델
├── human_pose_dataset.py        # COCO dataset loader
├── train_human_pose.py          # 학습 스크립트
├── train_human_pose.sh          # 학습 실행 shell 스크립트
└── HUMAN_POSE_README.md         # 이 문서
```

## COCO Keypoints (17개)

```python
0: nose
1: left_eye
2: right_eye
3: left_ear
4: right_ear
5: left_shoulder
6: right_shoulder
7: left_elbow
8: right_elbow
9: left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

## 데이터셋 준비

### COCO Dataset

1. COCO 2017 dataset 다운로드:
   ```bash
   # Images
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip

   # Annotations
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```

2. 압축 해제:
   ```bash
   unzip train2017.zip
   unzip val2017.zip
   unzip annotations_trainval2017.zip
   ```

3. 디렉토리 구조:
   ```
   /path/to/coco/
   ├── images/
   │   ├── train2017/
   │   └── val2017/
   └── annotations/
       ├── person_keypoints_train2017.json
       └── person_keypoints_val2017.json
   ```

## 학습 실행

### 1. Shell 스크립트 수정

`train_human_pose.sh` 파일을 열어서 데이터셋 경로를 수정하세요:

```bash
TRAIN_IMAGE_DIR="/path/to/coco/images/train2017"
TRAIN_ANNOTATION="/path/to/coco/annotations/person_keypoints_train2017.json"
VAL_IMAGE_DIR="/path/to/coco/images/val2017"
VAL_ANNOTATION="/path/to/coco/annotations/person_keypoints_val2017.json"
```

### 2. 학습 시작

```bash
chmod +x train_human_pose.sh
./train_human_pose.sh
```

### 3. 직접 Python 실행 (예시)

```bash
torchrun --nproc_per_node=4 train_human_pose.py \
    --train_image_dir /path/to/coco/images/train2017 \
    --train_annotation_file /path/to/coco/annotations/person_keypoints_train2017.json \
    --val_image_dir /path/to/coco/images/val2017 \
    --val_annotation_file /path/to/coco/annotations/person_keypoints_val2017.json \
    --model_name facebook/dinov2-base \
    --image_size 512 512 \
    --heatmap_size 512 512 \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 4 \
    --checkpoint_dir checkpoints_human_pose \
    --wandb_project DINOv3_HumanPose \
    --wandb_run_name human_pose_training
```

## 하이퍼파라미터 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_name` | `facebook/dinov2-base` | DINOv3 백본 모델 |
| `--image_size` | `512 512` | 입력 이미지 크기 (H, W) |
| `--heatmap_size` | `512 512` | 출력 heatmap 크기 (H, W) |
| `--lr` | `1e-4` | 기본 learning rate |
| `--batch_size` | `16` | GPU당 batch size |
| `--epochs` | `100` | 총 학습 에포크 |
| `--num_workers` | `4` | 데이터 로딩 워커 수 |

## 모델 선택

DINOv3 모델 옵션:
- `facebook/dinov2-small`: 작은 모델 (빠름, 정확도 낮음)
- `facebook/dinov2-base`: 기본 모델 (균형)
- `facebook/dinov2-large`: 큰 모델 (느림, 정확도 높음)
- `facebook/dinov2-giant`: 매우 큰 모델 (매우 느림, 최고 정확도)

## 학습 모니터링

### WandB

학습 중 WandB에서 다음 지표를 모니터링할 수 있습니다:
- `train_loss`: 학습 loss (weighted MSE)
- `val_loss`: 검증 loss
- `learning_rate`: 현재 learning rate

### Checkpoints

학습 중 두 개의 checkpoint가 저장됩니다:
- `checkpoints_human_pose/best_model.pth`: 최고 성능 모델
- `checkpoints_human_pose/latest_checkpoint.pth`: 가장 최근 모델 (학습 재개용)

## 모델 구조

```
DINOv3HumanPoseEstimator
├── DINOv3Backbone (frozen)
│   └── DINOv3 feature extraction
├── LightCNNStem
│   ├── Conv blocks (1/4 scale features)
│   └── Conv blocks (1/8 scale features)
└── HumanPoseKeypointHead
    ├── TokenFuser
    ├── UNet-style decoder with skip connections
    └── Heatmap predictor (17 channels)
```

## Loss Function

- **Weighted MSE Loss**: 각 keypoint의 visibility에 따라 가중치 적용
  - Visible keypoint: weight = 1.0
  - Occluded keypoint: weight = 0.5
  - Not labeled keypoint: weight = 0.0

## Data Augmentation

### Occlusion Augmentation
- Probability: 20%
- Random patches around keypoints
- 1-4 keypoints occluded per image
- Patch size: 5-15% of image size

## 다음 단계: 로봇 자세 추정과 통합

학습이 완료되면:

1. **Human pose head의 가중치 추출**
   ```python
   checkpoint = torch.load('checkpoints_human_pose/best_model.pth')
   human_head_weights = checkpoint['model_state_dict']
   ```

2. **통합 모델 생성**
   - `DINOv3PoseEstimator`에 `human_keypoint_head` 추가
   - 세 개의 head: robot keypoint, robot angles, human keypoint
   - 같은 DINOv3 feature 공유

3. **Multi-task 학습 (선택사항)**
   - Robot + Human 동시 학습
   - Loss balancing 필요

## 문제 해결

### CUDA Out of Memory
- `--batch_size` 줄이기
- `--image_size` 줄이기
- 더 작은 모델 사용 (`dinov2-small`)

### Dataset Loading Error
- 데이터셋 경로 확인
- COCO annotation 형식 확인
- 이미지 파일 존재 확인

### 학습이 수렴하지 않음
- Learning rate 조정 (`--lr`)
- Batch size 증가
- 더 많은 에포크 학습

## 참고 자료

- [COCO Dataset](https://cocodataset.org/)
- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [Human Pose Estimation Overview](https://paperswithcode.com/task/pose-estimation)
