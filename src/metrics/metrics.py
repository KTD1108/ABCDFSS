import torch
import numpy as np

class MetricTracker:
    """
    Tracks evaluation metrics:
    - mIoU: Class-averaged foreground Intersection over Union
    - FB-IoU: Foreground vs Background IoU
    """
    def __init__(self, class_ids: list = None):
        self.class_ids = list(class_ids) if class_ids is not None else []
        self.class_inter = {cid: 0.0 for cid in self.class_ids}
        self.class_union = {cid: 0.0 for cid in self.class_ids}
        
        self.total_fg_inter = 0.0
        self.total_fg_union = 0.0
        self.total_bg_inter = 0.0
        self.total_bg_union = 0.0

    @torch.no_grad()
    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, class_id: int):
        """
        pred_mask: [B, H, W] binary {0, 1}
        gt_mask: [B, H, W] binary {0, 1}
        class_id: int semantic class ID
        """
        pred = pred_mask.bool()
        gt = gt_mask.bool()

        # Foreground
        fg_inter = (pred & gt).sum().item()
        fg_union = (pred | gt).sum().item()

        # Background
        bg_inter = ((~pred) & (~gt)).sum().item()
        bg_union = ((~pred) | (~gt)).sum().item()

        # Update class statistics
        if class_id not in self.class_inter:
            self.class_ids.append(class_id)
            self.class_inter[class_id] = 0.0
            self.class_union[class_id] = 0.0

        self.class_inter[class_id] += fg_inter
        self.class_union[class_id] += fg_union

        # Update FB statistics
        self.total_fg_inter += fg_inter
        self.total_fg_union += fg_union
        self.total_bg_inter += bg_inter
        self.total_bg_union += bg_union

    def get_metrics(self):
        # Class-wise IoU
        class_ious = []
        for cid in self.class_ids:
            u = self.class_union[cid]
            if u > 0:
                class_ious.append(self.class_inter[cid] / u)

        miou = np.mean(class_ious) * 100.0 if len(class_ious) > 0 else 0.0

        # FB-IoU
        fg_iou = self.total_fg_inter / max(self.total_fg_union, 1e-6)
        bg_iou = self.total_bg_inter / max(self.total_bg_union, 1e-6)
        fb_iou = ((fg_iou + bg_iou) / 2.0) * 100.0

        return {
            'mIoU': miou,
            'FB-IoU': fb_iou,
            'class_ious': {cid: (self.class_inter[cid] / max(self.class_union[cid], 1e-6)) * 100.0 for cid in self.class_ids}
        }
