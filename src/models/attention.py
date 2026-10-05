import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCrossAttention(nn.Module):
    """
    Dense Cross-Attention mechanism computing fine-grained dense affinity maps
    between Query and Support feature representations.
    """
    def __init__(self, key_dim: int = 64):
        super(DenseCrossAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(key_dim)

    def forward(self, q_feat: torch.Tensor, s_feat: torch.Tensor, s_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_feat: [B, C, H_q, W_q] Query feature map
            s_feat: [B, K, C, H_s, W_s] Support feature maps across K shots
            s_mask: [B, K, H_img, W_img] Support binary ground-truth masks
        Returns:
            q_coarse: [B, H_q, W_q] Continuous coarse prediction map in [0, 1]
        """
        B, C, Hq, Wq = q_feat.shape
        _, K, _, Hs, Ws = s_feat.shape

        # Concatenate K support shots along spatial width: [B, C, Hs, Ws * K]
        s_feat_concat = torch.cat(s_feat.unbind(dim=1), dim=-1)
        
        # Downsample support masks to match support feature spatial size (matching author segutils.downsample_mask)
        s_masks_down = [
            F.interpolate(m.unsqueeze(1).float(), size=(Hs, Ws), mode='bilinear', align_corners=False).squeeze(1)
            for m in s_mask.unbind(dim=1)
        ]
        s_mask_concat = torch.cat(s_masks_down, dim=-1)  # [B, Hs, Ws * K]

        # Reshape to pixel matrices: Q = [B, Hq*Wq, C], K = [B, C, Hs*Ws*K]
        Q = q_feat.permute(0, 2, 3, 1).reshape(B, Hq * Wq, C)
        K_mat = s_feat_concat.permute(0, 2, 3, 1).reshape(B, Hs * Ws * K, C).transpose(1, 2)

        # Scaled dot-product affinity matrix: [B, Hq*Wq, Hs*Ws*K]
        affinity = torch.matmul(Q, K_mat) * self.scale
        affinity = F.softmax(affinity, dim=-1)

        # Filter affinity with binary support mask V: [B, Hs*Ws*K, 1]
        V = s_mask_concat.reshape(B, Hs * Ws * K, 1).float()
        q_coarse = torch.matmul(affinity, V).reshape(B, Hq, Wq)

        return q_coarse
