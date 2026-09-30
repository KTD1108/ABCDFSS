import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxWeightedFusion(nn.Module):
    """
    Task-Adaptive Softmax-Weighted Layer Fusion.
    Dynamically computes the discriminative margin delta_l between support foreground
    and background representations for each pyramid layer, allocating higher fusion
    weights to layers with sharper domain separation.
    """
    def __init__(self, num_layers: int = 13, temperature: float = 1.0):
        super(SoftmaxWeightedFusion, self).__init__()
        self.num_layers = num_layers
        self.temperature = max(float(temperature), 1e-3)

    @torch.no_grad()
    def compute_layer_margins(self, s_feats_adapted: list, s_mask: torch.Tensor, l0: int = 3) -> torch.Tensor:
        """
        Calculates discriminative margin delta_l for each layer l >= l0.
        delta_l = 1 - CosineSimilarity(fg_prototype, bg_prototype)
        """
        margins = []
        for sft in s_feats_adapted[l0:]:
            # sft: [B, K, C, Hs, Ws]
            B, K, C, Hs, Ws = sft.shape
            sft_flat = sft.reshape(B * K, C, Hs * Ws)

            # Downsample support mask to match feature map resolution
            smask_down = [
                F.interpolate(m.unsqueeze(1).float(), size=(Hs, Ws), mode='nearest').squeeze(1)
                for m in s_mask.unbind(dim=1)
            ]
            smask_cat = torch.stack(smask_down, dim=1).reshape(B * K, 1, Hs * Ws)

            fg_mask = smask_cat.float()
            bg_mask = 1.0 - fg_mask

            fg_count = fg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            bg_count = bg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            # Support prototypes
            fg_proto = (sft_flat * fg_mask).sum(dim=-1, keepdim=True) / fg_count
            bg_proto = (sft_flat * bg_mask).sum(dim=-1, keepdim=True) / bg_count

            fg_proto_norm = F.normalize(fg_proto, p=2, dim=1)
            bg_proto_norm = F.normalize(bg_proto, p=2, dim=1)

            # Cosine similarity between foreground and background prototypes
            cos_sim = (fg_proto_norm * bg_proto_norm).sum(dim=1).mean()
            margin = 1.0 - cos_sim
            margins.append(margin)

        return torch.stack(margins)

    def forward(self, q_coarses_per_layer: torch.Tensor, s_feats_adapted: list = None, s_mask: torch.Tensor = None, l0: int = 3) -> torch.Tensor:
        """
        Args:
            q_coarses_per_layer: [B, L, H_target, W_target] Stacked coarse predictions from all L layers
            s_feats_adapted: List of adapted feature tensors
            s_mask: Support ground-truth masks [B, K, H, W]
            l0: Starting layer index (default 3)
        Returns:
            q_fused: [B, H_target, W_target] Softmax-weighted fused prediction map
        """
        B, L, H, W = q_coarses_per_layer.shape

        if s_feats_adapted is not None and s_mask is not None:
            margins = self.compute_layer_margins(s_feats_adapted, s_mask, l0=l0)
            margins = margins.to(q_coarses_per_layer.device)
            weights = F.softmax(margins / self.temperature, dim=0)
        else:
            # Fallback to uniform weighting
            weights = torch.full((L,), 1.0 / L, device=q_coarses_per_layer.device)

        q_fused = (q_coarses_per_layer * weights.view(1, L, 1, 1)).sum(dim=1)
        return q_fused


class UniformFusion(nn.Module):
    """
    Standard flat average fusion: q_fused = 1/L * sum(q_coarse_l).
    """
    def forward(self, q_coarses_per_layer: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return q_coarses_per_layer.mean(dim=1)


def build_fusion(fusion_mode: str, num_layers: int = 13, temperature: float = 1.0) -> nn.Module:
    """Factory helper to build multi-layer fusion module."""
    mode = fusion_mode.lower()
    if mode in ['softmax_margin', 'softmax', 'margin']:
        return SoftmaxWeightedFusion(num_layers=num_layers, temperature=temperature)
    elif mode in ['mean', 'average', 'uniform']:
        return UniformFusion()
    else:
        raise ValueError(f"Unknown fusion mode: '{fusion_mode}'. Available: ['softmax_margin', 'mean']")
