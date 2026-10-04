import copy
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
                adapted_2d = F.normalize(adapter(feat_2d), p=2, dim=1)
                adapted = adapted_2d.view(B, K, self.out_channels, H, W)
            else:
                adapted = F.normalize(adapter(feat), p=2, dim=1)
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
        lr: float = 1e-2
    ):
        """
        Fits a single layer adapter using SGD optimization on InfoNCE + Variance regularization.
        """
        adapter = self.adapters[layer_idx]
        adapter.train()
        optimizer = torch.optim.SGD(adapter.parameters(), lr=lr)

        # q_feat: [B, C, H, W], q_feat_aug: [B, aug, C, H, W]
        B, aug, C, H, W = q_feat_aug.shape
        q_orig_expanded = q_feat.unsqueeze(1).expand(B, aug, C, H, W).reshape(B * aug, C, H, W)
        q_aug_flat = q_feat_aug.reshape(B * aug, C, H, W)

        # s_feat: [B, K, C, H, W], s_feat_aug: [B, K, aug, C, H, W]
        _, K, _, _, _ = s_feat.shape
        s_orig_expanded = s_feat.unsqueeze(2).expand(B, K, aug, C, H, W).reshape(B * K * aug, C, H, W)
        s_aug_flat = s_feat_aug.reshape(B * K * aug, C, H, W)

        with torch.enable_grad():
            for _ in range(num_epochs):
                optimizer.zero_grad()

                # Query forward pass
                q_orig_adapted = F.normalize(adapter(q_orig_expanded), p=2, dim=1)
                q_aug_adapted = F.normalize(adapter(q_aug_flat), p=2, dim=1)
                loss_q = self.info_nce_loss(q_orig_adapted, q_aug_adapted) + self.keep_var_loss(q_orig_adapted, q_aug_adapted)

                # Support forward pass
                s_orig_adapted = F.normalize(adapter(s_orig_expanded), p=2, dim=1)
                s_aug_adapted = F.normalize(adapter(s_aug_flat), p=2, dim=1)
                loss_s = self.info_nce_loss(s_orig_adapted, s_aug_adapted) + self.keep_var_loss(s_orig_adapted, s_aug_adapted)

                total_loss = loss_q + loss_s
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
        lr: float = 1e-2
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
            self.fit_layer(
                layer_idx=l_rel,
                q_feat=qf,
                s_feat=sf,
                q_feat_aug=qf_aug,
                s_feat_aug=sf_aug,
                s_mask_aug=s_masks_aug,
                num_epochs=num_epochs,
                lr=lr
            )
