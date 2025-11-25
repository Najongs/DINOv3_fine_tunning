import argparse
import time
from typing import List, Tuple

import torch

from model import DINOv3PoseEstimator
from dataset import IMAGE_RESOLUTION, HEATMAP_SIZE

def parse_resolution(text: str) -> Tuple[int, int]:
    """Parse strings like '640x480' (W×H) into (height, width)."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"잘못된 해상도 형식입니다: {text}")
    width, height = map(int, parts)
    return height, width


def parse_combo(text: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Parse strings like '640x640:320x320' into
    ((img_w, img_h), (heatmap_w, heatmap_h)).
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"콤보 문자열은 image:heatmap 형식을 따라야 합니다: {text}")
    return parse_resolution(parts[0]), parse_resolution(parts[1])


def resolve_model_name(model_type: str) -> str:
    if "vit" in model_type:
        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
    if "conv" in model_type:
        return "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    if "siglip2" in model_type:
        return "google/siglip2-base-patch16-224"
    if "siglip" in model_type:
        return "google/siglip-base-patch16-224"
    return "facebook/dinov3-vitb16-pretrain-lvd1689m"


def load_model(model_type: str, checkpoint_path: str, device: torch.device) -> DINOv3PoseEstimator:
    model_name = resolve_model_name(model_type)
    model = DINOv3PoseEstimator(dino_model_name=model_name, heatmap_size=HEATMAP_SIZE, ablation_mode=model_type)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        new_state = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state[new_key] = value
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[경고] 체크포인트에 없는 파라미터: {missing}")
        if unexpected:
            print(f"[경고] 모델에 없는 파라미터: {unexpected}")

    model.to(device)
    model.eval()
    return model


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_combo(
    model: DINOv3PoseEstimator,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    """Return average latency (ms) for a single (image, heatmap) combo."""
    height, width = image_size
    hm_h, hm_w = heatmap_size
    if hasattr(model, "keypoint_head"):
        model.keypoint_head.heatmap_size = (hm_h, hm_w)  # stored as (H, W)

    dummy = torch.randn(1, 3, height, width, device=device)

    with torch.no_grad():
        for _ in range(max(1, warmup)):
            model(dummy)
    synchronize(device)

    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy)
    synchronize(device)
    elapsed = (time.time() - start) * 1000.0 / max(1, iterations)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="입력/히트맵 해상도 조합에 따른 추론 지연 시간을 비교합니다."
    )
    parser.add_argument(
        "--model-type",
        default="dino_conv_only",
        help="Single_view_3D_Loss.py에서 사용하는 ablation_mode 값 (예: dino_conv_only, siglip_only 등)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="학습된 체크포인트 경로 (미지정 시 무작위 초기화 상태로 측정)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda 또는 cpu",
    )
    parser.add_argument(
        "--fallback-device",
        default="",
        help="주 장치에서 실패할 경우 사용할 예비 장치 (예: cpu). 비활성화하려면 빈 문자열 유지.",
    )
    parser.add_argument(
        "--combo",
        action="append",
        type=parse_combo,
        help="비교할 image:heatmap 해상도 (예: 640x640:320x320, 형식은 W×H). 여러 번 지정 가능.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="워밍업 반복 횟수")
    parser.add_argument("--iters", type=int, default=10, help="측정 반복 횟수")
    args = parser.parse_args()

    combos: List[Tuple[Tuple[int, int], Tuple[int, int]]] = (
        args.combo
        if args.combo
        else [
            (IMAGE_RESOLUTION, HEATMAP_SIZE),
            ((512, 512), (224, 224)),
            ((640, 640), (320, 320)),
            ((672, 672), (336, 336)),
        ]
    )

    def get_device(device_name: str) -> torch.device:
        try:
            return torch.device(device_name)
        except (RuntimeError, AssertionError) as exc:
            raise SystemExit(f"[오류] 사용할 수 없는 장치입니다: {device_name} ({exc})")

    primary_device = get_device(args.device)
    fallback_device = get_device(args.fallback_device) if args.fallback_device else None

    def is_device_available(device: torch.device) -> bool:
        if device.type == "cuda":
            return torch.cuda.is_available()
        return True

    if not is_device_available(primary_device):
        warning = f"[경고] 요청된 기본 장치 '{primary_device}' 를 사용할 수 없습니다."
        if fallback_device:
            if is_device_available(fallback_device):
                print(f"{warning} '{fallback_device}' 로 대체합니다.")
                primary_device, fallback_device = fallback_device, None
            else:
                print(f"{warning} 지정한 예비 장치 '{fallback_device}' 또한 사용할 수 없어 CPU로 전환합니다.")
                primary_device = torch.device("cpu")
                fallback_device = None
        else:
            raise SystemExit(
                f"{warning} --fallback-device 옵션을 사용해 다른 장치를 지정하거나 CUDA 환경을 확인하세요."
            )

    print(f"[정보] 기본 장치: {primary_device}, 모델 타입: {args.model_type}")
    if fallback_device:
        print(f"[정보] 예비 장치: {fallback_device}")
    if args.checkpoint:
        print(f"[정보] 체크포인트 로드: {args.checkpoint}")
    else:
        print("[정보] 체크포인트 없이 무작위 가중치로 측정합니다.")

    models = {}

    def ensure_model(device: torch.device) -> DINOv3PoseEstimator:
        key = str(device)
        if key not in models:
            print(f"[정보] 모델을 '{device}' 장치에 로드합니다.")
            models[key] = load_model(args.model_type, args.checkpoint, device)
        return models[key]

    print("\n조합별 지연시간 (ms):")
    for (img_h, img_w), (hm_h, hm_w) in combos:
        try:
            model = ensure_model(primary_device)
            latency = benchmark_combo(
                model,
                (img_h, img_w),
                (hm_h, hm_w),
                primary_device,
                args.warmup,
                args.iters,
            )
            print(f"  입력 {img_w}x{img_h}, 히트맵 {hm_w}x{hm_h} -> {latency:.2f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f"[오류] 입력 {img_w}x{img_h} 처리 중 CUDA 메모리 부족. 이 조합을 건너뜁니다.")
        except torch.AcceleratorError as e:
            print(f"[오류] 입력 {img_w}x{img_h} 처리 중 CUDA 오류 발생: {e}. 이 조합을 건너뜁니다.")
        except RuntimeError as err:
            if not fallback_device:
                raise
            print(
                f"[경고] {primary_device} 장치에서 '{img_w}x{img_h}' 조합 실행 실패: {err}. "
                f"{fallback_device} 장치로 재시도합니다."
            )
            try:
                model = ensure_model(fallback_device)
                latency = benchmark_combo(
                    model,
                    (img_h, img_w),
                    (hm_h, hm_w),
                    fallback_device,
                    args.warmup,
                    args.iters,
                )
                print(
                    f"  입력 {img_w}x{img_h}, 히트맵 {hm_w}x{hm_h} -> {latency:.2f} ms "
                    f"(fallback: {fallback_device})"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[오류] Fallback 장치에서 입력 {img_w}x{img_h} 처리 중 CUDA 메모리 부족. 이 조합을 건너뜁니다.")
            except torch.AcceleratorError as e:
                print(f"[오류] Fallback 장치에서 입력 {img_w}x{img_h} 처리 중 CUDA 오류 발생: {e}. 이 조합을 건너뜁니다.")



if __name__ == "__main__":
    main()
