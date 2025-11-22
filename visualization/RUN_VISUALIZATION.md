# 시각화 실행 가이드

## 빠른 시작

```bash
cd /home/najo/NAS/DIP/DINOv3_fine_tunning/visualization
./Visualization.sh
```

## 사용법

### 1. 기본 실행 (기본 체크포인트 사용)

```bash
./Visualization.sh
```

기본 체크포인트: `../checkpoints_total_dino_conv_only/best_model.pth`

### 2. 특정 체크포인트 사용

```bash
./Visualization.sh /path/to/your/checkpoint.pth
```

예시:
```bash
./Visualization.sh ../checkpoints_total_dino_vit_only/best_model.pth
```

## 출력 구조

스크립트는 다음과 같이 4개의 시각화 파일을 생성합니다:

```
visualization/results/
├── fr5_visualization_20250122_143022.png
├── franka_research3_visualization_20250122_143022.png
├── meca500_visualization_20250122_143022.png
└── meca_insertion_visualization_20250122_143022.png
```

파일명에는 타임스탬프가 포함되어 이전 결과를 덮어쓰지 않습니다.

## 실행 순서

1. **Fr5 로봇** (6관절, 3뷰 × 2카메라)
2. **Franka Research 3** (7관절, 4뷰 × 2카메라 × 2포즈)
3. **Meca500** (6관절, 단일 카메라, 6샘플)
4. **Meca Insertion** (6관절, 3뷰 × 2카메라)

## 시각화 내용

각 이미지는 다음을 포함합니다:

- 🟢 **녹색 선 (H0~H6)**: 히트맵 기반 2D 키포인트 예측
- 🟣 **마젠타 선 (J0~J6)**: FK 기반 3D 키포인트 투영

## 문제 해결

### 권한 오류
```bash
chmod +x Visualization.sh
```

### CUDA 메모리 부족
스크립트는 순차적으로 실행되므로 GPU 메모리 문제가 적습니다.
문제 발생 시 각 스크립트를 개별 실행하세요.

### 체크포인트 없음
```bash
# 체크포인트 경로 확인
ls -lh ../checkpoints_total_dino_conv_only/best_model.pth

# 다른 체크포인트 사용
./Visualization.sh ../checkpoints_total_dino_conv_only/latest_checkpoint.pth
```

## 개별 실행

특정 로봇만 시각화하려면:

```bash
# Fr5만
python visualize_fr5.py --output results/fr5_test.png

# Franka Research 3만
python visualize_franka_research3.py --output results/fr3_test.png

# Meca500만 (샘플 개수 지정)
python visualize_meca500.py --num_samples 9 --output results/meca_test.png

# Meca Insertion만
python visualize_meca_insertion.py --output results/meca_ins_test.png
```

## 결과 확인

```bash
# 생성된 파일 목록 확인
ls -lh results/

# 최신 파일 확인
ls -lt results/ | head -5

# 이미지 뷰어로 열기 (예시)
eog results/fr5_visualization_*.png  # GNOME
feh results/  # feh
```

## 성능 참고사항

- 각 스크립트는 약 10-30초 소요 (GPU 사용 시)
- 전체 실행 시간: 약 1-2분
- 생성되는 각 이미지 크기: 약 2-5MB
