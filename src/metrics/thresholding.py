import torch
import cv2
import numpy as np

def compute_otsu_threshold(prob_map: torch.Tensor, drop_least: float = 0.05) -> float:
    """
    Otsu thresholding matching paper implementation (segutils.otsus).
    Scales to uint8 [0, 255], truncates lowest drop_least percentiles,
    and applies cv2.THRESH_OTSU.
    """
    numpy_image = prob_map.detach().cpu().numpy().astype(np.float32)
    npmin, npmax = float(numpy_image.min()), float(numpy_image.max())
    if npmax - npmin < 1e-6:
        return (npmin + npmax) / 2.0

    normed = ((numpy_image - npmin) / (npmax - npmin + 1e-8) * 255.0).astype(np.uint8)
    truncated_vals = normed[normed >= int(255 * drop_least)]

    if len(truncated_vals) == 0:
        thresh_value = 128
    else:
        thresh_value, _ = cv2.threshold(truncated_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denorm_thresh = float(thresh_value / 255.0 * (npmax - npmin) + npmin)
    return denorm_thresh


def apply_adaptive_threshold(
    logit_mask: torch.Tensor,
    support_mask: torch.Tensor = None,
    method: str = 'pred_mean',
    s_mask: torch.Tensor = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes threshold and binary prediction mask.
    Matches paper protocol (runner.calcthresh):
        method == 'pred_mean': thresh = max(otsu_thresh, fused_pred.mean())
    Args:
        logit_mask: [B, H, W] Continuous fused prediction map
        support_mask (or s_mask): [B, K, H, W] Optional support ground truth mask
    Returns:
        thresholds: [B] Optimal threshold value
        pred_mask: [B, H, W] Binary segmentation mask {0, 1}
    """
    if support_mask is None and s_mask is not None:
        support_mask = s_mask
    B = logit_mask.shape[0]
    pred_masks = []
    thresholds = []

    for b in range(B):
        sample_logits = logit_mask[b]
        if method == 'pred_mean':
            otsu_th = compute_otsu_threshold(sample_logits, drop_least=0.05)
            mean_val = sample_logits.mean().item()
            th = max(otsu_th, mean_val)
        elif method == 'otsu':
            th = compute_otsu_threshold(sample_logits, drop_least=0.05)
        elif method == 'raw_mean':
            th = sample_logits.mean().item()
        else:
            th = 0.5

        pred = (sample_logits > th).float()
        thresholds.append(th)
        pred_masks.append(pred)

    return torch.tensor(thresholds, device=logit_mask.device), torch.stack(pred_masks, dim=0)
