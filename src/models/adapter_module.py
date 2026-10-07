import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import build_adapter
from .loss import DenseInfoNCELoss, KeepVarianceLoss, ContrastivePrototypeLoss

class TaskAdaptedHead(nn.Module):
    """
    Multi-layer adapter container managing attached adapters for ResNet-50 bottleneck layers.
    Executes test-time contrastive adaptation per task/class.
    """
    def __init__(
        self,
        adapter_type: str = 'depthwise_separable_3x3',
        out_channels: int = 64,
        l0: int = 3,
        num_layers: int = 16
    ):
        super(TaskAdaptedHead, self).__init__()
        self.adapter_type = adapter_type
        self.out_channels = out_channels
        self.l0 = l0
        self.num_layers = num_layers

        # Standard ResNet-50 channels across 16 bottleneck layers:
        # Layer 1: 3 blocks of 256
        # Layer 2: 4 blocks of 512
        # Layer 3: 6 blocks of 1024
        # Layer 4: 3 blocks of 2048
        self.in_channels_list = (
            [256] * 3 +
            [512] * 4 +
            [1024] * 6 +
            [2048] * 3
        )

        self.adapters = nn.ModuleList([
            build_adapter(adapter_type, in_c, out_channels)
            for in_c in self.in_channels_list[self.l0:]
        ])

        self.info_nce_loss = DenseInfoNCELoss(temperature=0.5)
        self.keep_var_loss = KeepVarianceLoss()
        self.proto_loss = ContrastivePrototypeLoss()

    def get_adapted_features(self, features_pyramid: list) -> list:
        """
        Passes backbone features through their respective attached adapters.
        Returns full pyramid with non-adapted layers kept as None or raw.
        """
        adapted_pyramid = [None] * self.l0
        for adapter, feat in zip(self.adapters, features_pyramid[self.l0:]):
            input_shape = feat.shape
            # If shape is [B, K, C, H, W], flatten to [B*K, C, H, W] for convolution
            if feat.dim() == 5:
                B, K, C, H, W = input_shape
                feat_2d = feat.view(B * K, C, H, W)
                adapted_2d = adapter(feat_2d)
                adapted = adapted_2d.view(B, K, self.out_channels, H, W)
            else:
                adapted = adapter(feat)
            adapted_pyramid.append(adapted)
        return adapted_pyramid

    def fit_layer(
        self,
        layer_idx: int,
        q_feat: torch.Tensor,
        s_feat: torch.Tensor,
        q_feat_aug: torch.Tensor,
        s_feat_aug: torch.Tensor,
        s_mask_aug: torch.Tensor,
        num_epochs: int = 25,
        lr: float = 1e-2,
        mapped_q_feat: torch.Tensor = None,
        mapped_s_feat: torch.Tensor = None
    ):
        """
        Fits a single layer adapter using SGD optimization on InfoNCE + Variance regularization + Prototype loss.
        """
        adapter = self.adapters[layer_idx]
        adapter.train()
        optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)

        # q_feat: [B, C, H, W], q_feat_aug: [B, aug, C, H, W]
        B, aug, C, H, W = q_feat_aug.shape
        if mapped_q_feat is not None:
            q_orig_input = mapped_q_feat.reshape(B * aug, C, H, W)
        else:
            q_orig_input = q_feat.unsqueeze(1).expand(B, aug, C, H, W).reshape(B * aug, C, H, W)
        q_aug_flat = q_feat_aug.reshape(B * aug, C, H, W)

        # s_feat: [B, K, C, H, W], s_feat_aug: [B, K, aug, C, H, W]
        _, K, _, _, _ = s_feat.shape
        if mapped_s_feat is not None:
            s_orig_input = mapped_s_feat.reshape(B * K * aug, C, H, W)
        else:
            s_orig_input = s_feat.unsqueeze(2).expand(B, K, aug, C, H, W).reshape(B * K * aug, C, H, W)
        s_aug_flat = s_feat_aug.reshape(B * K * aug, C, H, W)

        with torch.enable_grad():
            for _ in range(num_epochs):
                optimizer.zero_grad()

                # Query forward pass
                q_orig_adapted = F.normalize(adapter(q_orig_input), p=2, dim=1)
                q_aug_adapted = F.normalize(adapter(q_aug_flat), p=2, dim=1)
                loss_q = self.info_nce_loss(q_orig_adapted, q_aug_adapted) + self.keep_var_loss(q_orig_adapted, q_aug_adapted)

                # Support forward pass
                s_orig_adapted = F.normalize(adapter(s_orig_input), p=2, dim=1)
                s_aug_adapted = F.normalize(adapter(s_aug_flat), p=2, dim=1)
                loss_s = self.info_nce_loss(s_orig_adapted, s_aug_adapted) + self.keep_var_loss(s_orig_adapted, s_aug_adapted)

                # Contrastive Prototype Loss (Eq. 4 in CVPR 2024 paper)
                if s_mask_aug is not None:
                    B_s, K_s, aug_s, H_m, W_m = s_mask_aug.shape
                    m_down = F.interpolate(
                        s_mask_aug.view(B_s * K_s * aug_s, 1, H_m, W_m).float(),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                    proto_base = s_orig_adapted.view(K_s, aug_s, self.out_channels, H * W).permute(1, 2, 0, 3).reshape(aug_s, self.out_channels, K_s * H * W)
                    proto_trans = s_aug_adapted.view(K_s, aug_s, self.out_channels, H * W).permute(1, 2, 0, 3).reshape(aug_s, self.out_channels, K_s * H * W)
                    mask_flat = m_down.view(K_s, aug_s, H * W).permute(1, 0, 2).reshape(aug_s, 1, K_s * H * W)

                    fg_p_base = (proto_base * mask_flat).sum(dim=-1) / (mask_flat.sum(dim=-1) + 1e-8)
                    fg_p_trans = (proto_trans * mask_flat).sum(dim=-1) / (mask_flat.sum(dim=-1) + 1e-8)
                    bg_p_trans = (proto_trans * (1.0 - mask_flat)).sum(dim=-1) / ((1.0 - mask_flat).sum(dim=-1) + 1e-8)

                    sim_fg_fg = F.cosine_similarity(fg_p_base, fg_p_trans, dim=-1)
                    sim_fg_bg = F.cosine_similarity(fg_p_base, bg_p_trans, dim=-1)
                    loss_proto = -torch.log(torch.exp(sim_fg_fg) / (torch.exp(sim_fg_fg) + torch.exp(sim_fg_bg) + 1e-8)).mean()
                else:
                    loss_proto = 0.0

                total_loss = loss_q + loss_s + loss_proto
                total_loss.backward()
                optimizer.step()

        adapter.eval()

    def fit(
        self,
        q_feats: list,
        s_feats: list,
        q_feats_aug: list,
        s_feats_aug: list,
        s_masks_aug: torch.Tensor,
        num_epochs: int = 25,
        lr: float = 1e-2,
        mapped_q_feats: list = None,
        mapped_s_feats: list = None
    ):
        """
        Runs SGD adaptation loop across all L layers.
        """
        for l_rel, (qf, sf, qf_aug, sf_aug) in enumerate(zip(
            q_feats[self.l0:],
            s_feats[self.l0:],
            q_feats_aug[self.l0:],
            s_feats_aug[self.l0:]
        )):
            mqf = mapped_q_feats[l_rel] if mapped_q_feats is not None else None
            msf = mapped_s_feats[l_rel] if mapped_s_feats is not None else None
            self.fit_layer(
                layer_idx=l_rel,
                q_feat=qf,
                s_feat=sf,
                q_feat_aug=qf_aug,
                s_feat_aug=sf_aug,
                s_mask_aug=s_masks_aug,
                num_epochs=num_epochs,
                lr=lr,
                mapped_q_feat=mqf,
                mapped_s_feat=msf
            )
