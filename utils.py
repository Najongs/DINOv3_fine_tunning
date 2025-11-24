import os
import numpy as np
import torch
import torch.distributed as torch_dist
import cv2

# ======================= DDP 설정 함수 =======================
def setup_ddp():
    """DDP 프로세스 그룹을 초기화합니다."""
    torch_dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    """DDP 프로세스 그룹을 정리합니다."""
    torch_dist.destroy_process_group()

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_max_preds(batch_heatmaps):
    """heatmap에서 최대값의 좌표를 추출합니다."""
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_heatmaps should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape(batch_size, num_joints, 1)
    idx = idx.reshape(batch_size, num_joints, 1)

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def select_valid_correspondences(joints_3d_robot, kpts2d_padded, visibility_mask=None):
    joints_3d_robot = _to_np(joints_3d_robot).astype(np.float64)
    kpts2d_padded   = _to_np(kpts2d_padded).astype(np.float64)
    valid2d_mask = np.linalg.norm(kpts2d_padded, axis=1) > 0.0
    if visibility_mask is not None:
        vis_mask = _to_np(visibility_mask).astype(bool)
        max_len = min(len(valid2d_mask), len(vis_mask))
        combined_mask = np.zeros_like(valid2d_mask, dtype=bool)
        combined_mask[:max_len] = np.logical_and(valid2d_mask[:max_len], vis_mask[:max_len])
        valid2d_mask = combined_mask
    idx2d = np.where(valid2d_mask)[0]

    if idx2d.size == 0:
        raise ValueError("No valid 2D keypoints found.")

    # 로봇 FK로 나온 관절 개수와 2D 유효 개수의 최소치만 사용
    n = int(min(len(joints_3d_robot), idx2d.size))
    objp = joints_3d_robot[:n]          # (n,3)
    imgp = kpts2d_padded[idx2d[:n]]     # (n,2)

    if n < 4:
        raise ValueError(f"Need at least 4 correspondences for PnP, got {n}.")
    return objp, imgp

def solve_pnp_from_fk(joints_3d_robot, kpts2d_padded, K, dist, visibility_mask=None):
    objp, imgp = select_valid_correspondences(joints_3d_robot, kpts2d_padded, visibility_mask)
    K   = _to_np(K).astype(np.float64)
    dist= _to_np(dist).astype(np.float64).reshape(-1,)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_EPNP)
    rvec = rvec.reshape(3)
    tvec = tvec.reshape(3)
    return rvec, tvec, bool(ok)

def transform_robot_to_camera(joints_3d_robot, rvec, tvec):
    joints_3d_robot = _to_np(joints_3d_robot).astype(np.float64)
    R,_ = cv2.Rodrigues(rvec.reshape(3,1))
    t = tvec.reshape(3,1)
    X = joints_3d_robot.T  # (3,J)
    Y = (R @ X + t).T      # (J,3)
    return Y.astype(np.float32)

def compute_masked_loss(pred_heatmaps, gt_heatmaps,
                        pred_angles, gt_angles,
                        pred_3d, gt_3d,
                        joint_lengths, angle_lengths, point_lengths,
                        joint_confidences,
                        loss_fn_h, loss_fn_a, loss_fn_3D,
                        weight_h=1.0, weight_a=1.0, weight_3d=1.0):

    device = pred_heatmaps.device
    B, J, H, W = gt_heatmaps.shape
    A = gt_angles.shape[1]
    N = gt_3d.shape[1]
    
    # 1) Heatmap Loss (masked + confidence-aware)
    mask_h = (torch.arange(J, device=device)[None, :] < joint_lengths[:, None]).float()
    mask_h = mask_h[:, :, None, None].expand_as(gt_heatmaps)  # (B,J,H,W)
    if joint_confidences is not None:
        mask_h = mask_h * joint_confidences[:, :, None, None]

    l_h = loss_fn_h(pred_heatmaps, gt_heatmaps)  # (B,J,H,W)
    loss_h = (l_h * mask_h).sum() / mask_h.sum().clamp_min(1.0)

    # 2) Joint Angle Loss (masked + confidence-aware with joint visibility)
    mask_a = (torch.arange(A, device=device)[None, :] < angle_lengths[:, None]).float()
    if joint_confidences is not None:
        angle_conf = torch.ones_like(mask_a)
        max_transfer = min(joint_confidences.shape[1], A)
        angle_conf[:, :max_transfer] = joint_confidences[:, :max_transfer]
        mask_a = mask_a * angle_conf
    l_a = loss_fn_a(pred_angles, gt_angles)  # (B,A)
    loss_a = (l_a * mask_a).sum() / mask_a.sum().clamp_min(1.0)

    # 3) 3D Coordinate Loss (masked + confidence-aware)
    mask_p = (torch.arange(N, device=device)[None, :] < point_lengths[:, None]).float()
    if joint_confidences is not None:
        mask_p = mask_p * joint_confidences[:, :N]
    mask_p = mask_p[:, :, None].expand(-1, -1, 3)  # (B,N,3)

    l_3d = loss_fn_3D(pred_3d, gt_3d)  # (B,N,3) if reduction='none'
    loss_3d = (l_3d * mask_p).sum() / mask_p.sum().clamp_min(1.0)

    # 4) Weighted Sum
    total_loss = weight_h * loss_h + weight_a * loss_a + weight_3d * loss_3d

    return total_loss, {
        'loss_h': loss_h.detach(),
        'loss_a': loss_a.detach(),
        'loss_3d': loss_3d.detach()
    }

def save_checkpoints(checkpoint_data, best_model_state_dict, checkpoint_dir, is_best):
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
    if is_best:
        torch.save(best_model_state_dict, os.path.join(checkpoint_dir, "best_model.pth"))
