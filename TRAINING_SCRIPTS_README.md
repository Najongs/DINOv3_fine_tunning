# Depth Training Scripts

학습 스크립트 사용 가이드입니다.

## 📝 스크립트 목록

### 1. `train_depth.sh` (기본 - 권장)
**DINOv2-base 모델로 학습**

```bash
./train_depth.sh
```

**설정:**
- Model: `facebook/dinov2-base`
- Batch size: 16 per GPU (Total: 48)
- Learning rate: 1e-4
- Epochs: 50
- GPUs: 3 (RTX 3090)

**예상 시간:** ~15-20분/epoch, 총 12-17시간


### 2. `train_depth_background.sh` (백그라운드 실행)
**백그라운드에서 실행하고 로그 파일에 저장**

```bash
./train_depth_background.sh
```

학습이 백그라운드에서 실행되며, 로그가 `logs/` 폴더에 저장됩니다.

**모니터링:**
```bash
# 로그 실시간 확인
tail -f logs/train_depth_*.log

# GPU 사용률 확인
watch -n 1 nvidia-smi
```

**중단:**
```bash
# PID는 스크립트 실행 시 출력됨
kill <PID>
```


### 3. `train_depth_large.sh` (고품질)
**DINOv2-large 모델로 학습 - 더 높은 품질**

```bash
./train_depth_large.sh
```

**설정:**
- Model: `facebook/dinov2-large` (더 큰 모델)
- Batch size: 12 per GPU (Total: 36)
- Learning rate: 1e-4
- Epochs: 50

**장점:** 더 높은 depth 정확도
**단점:** 학습 시간 약 1.5배 증가


## 🔧 스크립트 수정 방법

스크립트를 직접 편집하여 설정을 변경할 수 있습니다:

```bash
vim train_depth.sh
```

### 주요 파라미터

```bash
# GPU 개수
export CUDA_VISIBLE_DEVICES=0,1,2  # 사용할 GPU ID
NUM_GPUS=3

# 학습 하이퍼파라미터
LR=1e-4           # Learning rate
BATCH_SIZE=16     # Per-GPU batch size
EPOCHS=50         # 총 epoch 수

# 모델 설정
MODEL_NAME="facebook/dinov2-base"  # 모델 선택
DEPTH_SIZE="280,504"               # Depth 출력 크기 (H,W)
RUN_NAME="depth_dinov2_base_v1"    # 실험 이름
```

### 사용 가능한 모델

```bash
# 작고 빠름
MODEL_NAME="facebook/dinov2-base"

# 크고 정확함
MODEL_NAME="facebook/dinov2-large"

# DINOv3 (최신)
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
```


## 📊 학습 모니터링

### Weights & Biases
학습 진행 상황은 자동으로 wandb에 로깅됩니다:
- Project: `DINOv3_Depth_Estimation`
- Run name: 스크립트에서 설정한 `RUN_NAME`

### Checkpoints
체크포인트는 다음 위치에 저장됩니다:
```
checkpoints_depth_{RUN_NAME}/
├── best_model.pth          # 최고 성능 모델
└── latest_checkpoint.pth   # 최신 체크포인트 (학습 재개용)
```


## 🔄 학습 재개

학습이 중단된 경우, 동일한 스크립트를 다시 실행하면 자동으로 재개됩니다:

```bash
# 같은 명령어로 재실행
./train_depth.sh
```

`latest_checkpoint.pth`가 존재하면 자동으로 로드됩니다.


## 💡 팁

### 1. GPU 메모리 부족 시
```bash
# train_depth.sh 수정
BATCH_SIZE=12  # 16 → 12로 감소
```

### 2. 더 빠른 학습
```bash
# Epoch 수 감소
EPOCHS=30  # 50 → 30

# 또는 더 작은 모델 사용
MODEL_NAME="facebook/dinov2-small"
```

### 3. 더 높은 해상도
```bash
# Depth 출력 크기 증가
DEPTH_SIZE="360,640"  # 280,504 → 360,640
```

### 4. Learning Rate Tuning
```bash
# 더 빠른 수렴 (불안정할 수 있음)
LR=2e-4

# 더 안정적인 학습 (느림)
LR=5e-5
```


## 🎯 추천 설정

### 빠른 프로토타입
```bash
MODEL_NAME="facebook/dinov2-base"
BATCH_SIZE=16
EPOCHS=30
LR=1e-4
```

### 최고 품질
```bash
MODEL_NAME="facebook/dinov2-large"
BATCH_SIZE=12
EPOCHS=50
LR=5e-5
```

### 밸런스 (권장)
```bash
MODEL_NAME="facebook/dinov2-base"
BATCH_SIZE=16
EPOCHS=50
LR=1e-4
```


## 📞 문제 해결

### CUDA Out of Memory
```bash
# Batch size 감소
BATCH_SIZE=8  # 또는 더 작게

# GPU 수 증가
NUM_GPUS=4
```

### Port already in use
```bash
# 다른 포트 사용
MASTER_PORT=29501  # 29500 → 29501
```

### 학습이 시작되지 않음
```bash
# Python 환경 확인
which python
python --version

# PyTorch 설치 확인
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```
