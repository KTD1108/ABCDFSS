import torch

def compute_otsu_threshold(prob_map: torch.Tensor, n_bins: int = 256) -> float:
    """
    Otsu thresholding directly computed on PyTorch tensor probability map.
    Finds the optimal threshold that minimizes intra-class variance.
    """
    flat = prob_map.float().view(-1)
    min_val, max_val = flat.min().item(), flat.max().item()
    if max_val - min_val < 1e-5:
        return (min_val + max_val) / 2.0

    # Histogram of probabilities
    hist = torch.histc(flat, bins=n_bins, min=min_val, max=max_val)
    prob = hist / hist.sum()

    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * torch.arange(n_bins, device=prob.device), dim=0)
    mu_t = mu[-1]

    # Between-class variance
    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-7)
    max_idx = torch.argmax(sigma_b_squared).item()

    thresh = min_val + (max_idx / n_bins) * (max_val - min_val)
    return float(thresh)


def apply_adaptive_threshold(
    logit_mask: torch.Tensor,
    support_mask: torch.Tensor = None,
    method: str = 'pred_mean',
    s_mask: torch.Tensor = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes threshold and binary prediction mask.
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
        if method == 'otsu':
            th = compute_otsu_threshold(sample_logits)
        elif method == 'pred_mean':
            th = sample_logits.mean().item()
        else:
            th = 0.5

        pred = (sample_logits > th).float()
        thresholds.append(th)
        pred_masks.append(pred)

    return torch.tensor(thresholds, device=logit_mask.device), torch.stack(pred_masks, dim=0)
