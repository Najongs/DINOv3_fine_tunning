import cv2
import numpy as np
import sys
from pathlib import Path

# Allow importing from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Single_view_3D_Loss import get_max_preds, HEATMAP_CONF_THRESHOLD


def decode_keypoints_with_confidence(pred_heatmaps, image_shape, threshold=HEATMAP_CONF_THRESHOLD):
    """
    Convert predicted heatmaps to image-space keypoints, confidences, and visibility mask.
    Returns (keypoints_np, confidences_np, visibility_bool).
    """
    pred_heatmaps_np = pred_heatmaps.detach().cpu().numpy()
    pred_kpts, confidences = get_max_preds(pred_heatmaps_np)
    pred_kpts = pred_kpts[0]
    confidences = confidences[0, :, 0]

    h, w = image_shape[:2]
    heatmap_h, heatmap_w = pred_heatmaps.shape[2], pred_heatmaps.shape[3]

    scaled_kpts = pred_kpts.copy()
    scaled_kpts[:, 0] *= (w / float(heatmap_w))
    scaled_kpts[:, 1] *= (h / float(heatmap_h))

    visibility = confidences >= threshold
    return scaled_kpts, confidences, visibility


def annotate_confidence_panel(image, confidences, visibility, origin=(10, 20), line_height=18):
    """
    Draw joint confidence/visibility info directly on the image.
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (conf, vis) in enumerate(zip(confidences, visibility)):
        status = "OK " if vis else "DROP"
        color = (0, 200, 0) if vis else (0, 0, 255)
        text = f"J{idx}: {conf:.2f} {status}"
        position = (origin[0], origin[1] + line_height * idx)
        cv2.putText(annotated, text, position, font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, text, position, font, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def filtered_joint_summary(confidences, visibility):
    """
    Return textual summary describing which joints were filtered out.
    """
    summary = []
    for idx, (conf, vis) in enumerate(zip(confidences, visibility)):
        state = "kept" if vis else "filtered"
        summary.append(f"J{idx}: {conf:.2f} ({state})")
    return ", ".join(summary)


def select_pnp_indices(confidences, visibility, min_points=6, prefer_points=8):
    """
    Select indices to use for PnP. Prefer visible joints but fall back to the
    highest-confidence joints if too few remain after masking.
    Returns (indices_list, used_fallback: bool).
    """
    total = len(confidences)
    prefer_points = min(prefer_points, total)
    min_points = min(min_points, total)

    visible_indices = [idx for idx, vis in enumerate(visibility) if vis]
    if len(visible_indices) >= min_points:
        selected = visible_indices[:prefer_points]
        return selected, False

    sorted_indices = np.argsort(-np.asarray(confidences))
    fallback_count = max(min_points, prefer_points)
    fallback_count = min(fallback_count, total)
    selected = sorted_indices[:fallback_count].tolist()

    if len(selected) < min_points:
        return [], True
    return selected, True
