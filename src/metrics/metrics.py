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
        self.observed_classes = set()
        self.episode_ious = []
        self.episode_details = []

    @torch.no_grad()
    def update(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, class_id: int, metadata: dict = None):
        """
        pred_mask: [B, H, W] binary {0, 1}
        gt_mask: [B, H, W] binary {0, 1}
        class_id: int semantic class ID
        metadata: optional dict containing episode_id, query_img, support_imgs, category
        """
        pred = pred_mask.bool()
        gt = gt_mask.bool()

        # Foreground
        fg_inter = (pred & gt).sum().item()
        fg_union = (pred | gt).sum().item()

        # Background
        bg_inter = ((~pred) & (~gt)).sum().item()
        bg_union = ((~pred) | (~gt)).sum().item()

        # Track per-episode IoU
        ep_iou = (fg_inter / fg_union) if fg_union > 0 else 1.0
        self.episode_ious.append(ep_iou)

        # Track detailed trace if available
        detail = {
            'episode_id': metadata.get('episode_id', len(self.episode_ious) - 1) if metadata else len(self.episode_ious) - 1,
            'class_id': class_id,
            'iou': round(ep_iou * 100.0, 4)
        }
        if metadata:
            if 'query_img' in metadata:
                detail['query_img'] = metadata['query_img']
            if 'support_imgs' in metadata:
                detail['support_imgs'] = metadata['support_imgs']
            if 'category' in metadata:
                detail['category'] = metadata['category']
        self.episode_details.append(detail)

        # Update class statistics
        self.observed_classes.add(class_id)
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
        # Class-wise IoU (only across classes that actually appeared)
        class_ious = []
        for cid in self.class_ids:
            if cid in self.observed_classes:
                u = self.class_union[cid]
                if u > 0:
                    class_ious.append(self.class_inter[cid] / u)

        miou = float(np.mean(class_ious) * 100.0) if len(class_ious) > 0 else 0.0
        mean_episode_iou = float(np.mean(self.episode_ious) * 100.0) if self.episode_ious else 0.0

        # FB-IoU
        fg_iou = self.total_fg_inter / max(self.total_fg_union, 1e-6)
        bg_iou = self.total_bg_inter / max(self.total_bg_union, 1e-6)
        fb_iou = float(((fg_iou + bg_iou) / 2.0) * 100.0)

        return {
            'mIoU': miou,
            'Cumulative_mIoU': miou,
            'Mean_Episode_IoU': mean_episode_iou,
            'FB-IoU': fb_iou,
            'class_ious': {
                cid: float((self.class_inter[cid] / max(self.class_union[cid], 1e-6)) * 100.0)
                for cid in self.class_ids
                if cid in self.observed_classes
            },
            'episode_ious': [round(x * 100.0, 2) for x in self.episode_ious],
            'detailed_episodes': self.episode_details
        }
