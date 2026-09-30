import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseInfoNCELoss(nn.Module):
    """
    Dense pixel-level InfoNCE contrastive loss enforcing correspondence
    between original and augmented feature representations.
    """
    def __init__(self, temperature: float = 0.5):
        super(DenseInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, original_feats: torch.Tensor, transformed_feats: torch.Tensor) -> torch.Tensor:
        """
        original_feats: [B, C, H, W]
        transformed_feats: [B, C, H, W]
        """
        B, C, H, W = transformed_feats.shape
        o_flat = original_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t_flat = transformed_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Positive pairs: corresponding spatial locations
        positive_logits = torch.einsum('bik,bik->bi', o_flat, t_flat) / self.temperature
        
        # All pairwise pairs: spatial cross-similarity matrix
        all_logits = torch.einsum('bik,bjk->bij', o_flat, t_flat) / self.temperature

        # Numerically stable Log-Sum-Exp
        max_logits = torch.max(all_logits, dim=-1, keepdim=True).values
        log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(all_logits - max_logits), dim=-1, keepdim=True))

        loss = -(positive_logits - log_sum_exp.squeeze(-1))
        return loss.mean()


class KeepVarianceLoss(nn.Module):
    """
    Feature distribution regularization matching spatial channel-wise mean
    and variance between original and augmented feature maps.
    """
    def forward(self, original_feats: torch.Tensor, transformed_feats: torch.Tensor) -> torch.Tensor:
        mean_diff = original_feats.mean(dim=(-2, -1)) - transformed_feats.mean(dim=(-2, -1))
        var_diff = original_feats.var(dim=(-2, -1)) - transformed_feats.var(dim=(-2, -1))
        return torch.abs(mean_diff).mean() + torch.abs(var_diff).mean()


class ContrastivePrototypeLoss(nn.Module):
    """
    Support foreground/background prototype alignment loss.
    Maximizes cosine similarity between original and transformed foreground prototypes
    while minimizing similarity to background prototypes.
    """
    def forward(self, fg_proto_orig: torch.Tensor, fg_proto_trans: torch.Tensor, bg_proto_trans: torch.Tensor) -> torch.Tensor:
        sim_fg_fg = F.cosine_similarity(fg_proto_orig, fg_proto_trans, dim=-1)
        sim_fg_bg = F.cosine_similarity(fg_proto_orig, bg_proto_trans, dim=-1)
        
        numerator = torch.exp(sim_fg_fg)
        denominator = numerator + torch.exp(sim_fg_bg)
        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()
