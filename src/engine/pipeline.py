import time
import torch
import torch.nn.functional as F

from ..models.backbone import ResNetBackbone
from ..models.adapter_module import TaskAdaptedHead
from ..models.attention import DenseCrossAttention
from ..models.fusion import build_fusion
from ..utils.augmentations import TaskAugmentator
from ..metrics.metrics import MetricTracker
from ..metrics.thresholding import apply_adaptive_threshold

class CDFSSEngine:
    """
    Unified, clean execution engine for Cross-Domain Few-Shot Segmentation.
    Implements:
    - Algorithm 2: Adapt and Infer (every-episode optimization)
    - Algorithm 3: Quick-Infer with Cached Task Adapters (first-episode per class)
    - Softmax-Weighted Layer Fusion & Depthwise Separable Adapters
    """
    def __init__(
        self,
        adapter_type: str = 'depthwise_separable_3x3',
        fusion_mode: str = 'softmax_margin',
        fusion_temp: float = 1.0,
        adapt_mode: str = 'first-episode',
        num_epochs: int = 25,
        lr: float = 1e-2,
        l0: int = 3,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        self.adapter_type = adapter_type
        self.fusion_mode = fusion_mode
        self.fusion_temp = fusion_temp
        self.adapt_mode = adapt_mode
        self.num_epochs = num_epochs
        self.lr = lr
        self.l0 = l0

        # Core modules
        self.backbone = ResNetBackbone().to(self.device)
        self.cross_attention = DenseCrossAttention(key_dim=64).to(self.device)
        self.fusion_module = build_fusion(fusion_mode, num_layers=16 - l0, temperature=fusion_temp).to(self.device)
        self.augmentator = TaskAugmentator(num_transforms=2, blur_kernel_size=1, max_shear=20)

        # Cache of task-adapted heads per class
        self.class_adapter_cache = {}

    def _get_or_fit_adapter(
        self,
        class_id: int,
        q_img: torch.Tensor,
        s_img: torch.Tensor,
        s_mask: torch.Tensor,
        q_feats: list,
        s_feats: list
    ) -> TaskAdaptedHead:
        """
        Retrieves cached adapter or adapts from scratch (Algorithm 2 vs Algorithm 3).
        """
        if self.adapt_mode == 'first-episode' and class_id in self.class_adapter_cache:
            return self.class_adapter_cache[class_id]

        # Initialize fresh adapter head for this task
        adapter_head = TaskAdaptedHead(
            adapter_type=self.adapter_type,
            out_channels=64,
            l0=self.l0,
            num_layers=16
        ).to(self.device)

        # Generate augmented views using consistent episode-level transforms
        self.augmentator.setup_transforms()
        q_aug = self.augmentator.augment(q_img, reuse_transforms=True)
        
        # s_img: [B, K, 3, H, W] -> treat B*K as batch
        B, K, C, H, W = s_img.shape
        s_img_flat = s_img.view(B * K, C, H, W)
        s_mask_flat = s_mask.view(B * K, H, W)
        s_aug_flat, s_mask_aug_flat = self.augmentator.augment(s_img_flat, s_mask_flat, reuse_transforms=True)
        
        aug = q_aug.shape[1]
        s_aug = s_aug_flat.view(B, K, aug, C, H, W)
        s_mask_aug = s_mask_aug_flat.view(B, K, aug, H, W)

        # Precompute affine-mapped features to maintain spatial correspondence (applyAffines)
        mapped_q_feats = [self.augmentator.apply_affines(f) for f in q_feats]
        mapped_s_feats = []
        for f in s_feats:
            B_f, K_f, C_f, H_f, W_f = f.shape
            m_s = torch.stack([self.augmentator.apply_affines(f[:, k]) for k in range(K_f)], dim=1)
            mapped_s_feats.append(m_s)

        # Extract features for augmented views
        with torch.no_grad():
            q_feats_aug = self.backbone.extract_features(q_aug.view(B * aug, C, H, W))
            q_feats_aug = [f.view(B, aug, *f.shape[1:]) for f in q_feats_aug]

            s_feats_aug = self.backbone.extract_features(s_aug.view(B * K * aug, C, H, W))
            s_feats_aug = [f.view(B, K, aug, *f.shape[1:]) for f in s_feats_aug]

        # Run test-time contrastive adaptation
        adapter_head.fit(
            q_feats=q_feats,
            s_feats=s_feats,
            q_feats_aug=q_feats_aug,
            s_feats_aug=s_feats_aug,
            s_masks_aug=s_mask_aug,
            num_epochs=self.num_epochs,
            lr=self.lr,
            mapped_q_feats=mapped_q_feats[self.l0:],
            mapped_s_feats=mapped_s_feats[self.l0:]
        )

        if self.adapt_mode == 'first-episode':
            self.class_adapter_cache[class_id] = adapter_head

        return adapter_head

    def evaluate_episode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Processes a single few-shot segmentation episode.
        """
        q_img = batch['query_img'].to(self.device)
        q_mask = batch['query_mask'].to(self.device)
        s_imgs, s_masks = batch['support_set']
        s_imgs = s_imgs.to(self.device)
        s_masks = s_masks.to(self.device)
        class_id = int(batch['class_id'].item() if torch.is_tensor(batch['class_id']) else batch['class_id'])

        B, C, H_img, W_img = q_img.shape
        _, K, _, _, _ = s_imgs.shape

        # 1. Extract base backbone features (no grad needed for backbone)
        with torch.no_grad():
            q_feats = self.backbone.extract_features(q_img)
            s_feats = self.backbone.extract_features(s_imgs.view(B * K, C, H_img, W_img))
            s_feats = [f.view(B, K, *f.shape[1:]) for f in s_feats]

        # 2. Get or adapt attached adapters (requires gradients for adapter weights)
        adapter_head = self._get_or_fit_adapter(class_id, q_img, s_imgs, s_masks, q_feats, s_feats)

        # 3. Evaluation inference (no grad needed)
        with torch.no_grad():
            # Project features through adapters
            q_feats_adapted = adapter_head.get_adapted_features(q_feats)
            s_feats_adapted = adapter_head.get_adapted_features(s_feats)

            # 4. Dense Cross-Attention per layer
            # Intermediate spatial scale of layer l0 (e.g. 50x50 for 400x400 input, matching core/denseaffinity.py)
            h0, w0 = q_feats_adapted[self.l0].shape[-2:]
            layer_predictions = []
            for l_idx in range(self.l0, 16):
                q_l = q_feats_adapted[l_idx]
                s_l = s_feats_adapted[l_idx]
                # Coarse query prediction [B, Hq, Wq]
                q_coarse = self.cross_attention(q_l, s_l, s_masks)
                # Upsample to base layer l0 resolution (50x50)
                q_coarse_up = F.interpolate(q_coarse.unsqueeze(1), size=(h0, w0), mode='bilinear', align_corners=False).squeeze(1)
                layer_predictions.append(q_coarse_up)

            # Stack predictions across all L layers: [B, L, h0, w0]
            q_coarses_stacked = torch.stack(layer_predictions, dim=1)

            # 5. Multi-layer fusion at intermediate feature scale
            q_fused_coarse = self.fusion_module(q_coarses_stacked, s_feats_adapted=s_feats_adapted, s_mask=s_masks, l0=self.l0)

            # Upsample fused prediction map to full image resolution
            q_fused = F.interpolate(q_fused_coarse.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(1)

            # 6. Adaptive Thresholding
            _, pred_mask = apply_adaptive_threshold(q_fused, support_mask=s_masks, method='pred_mean')

        return pred_mask, q_mask, class_id

    def evaluate_dataset(self, dataloader, benchmark_name: str = "dataset", max_episodes: int = None):
        """
        Runs complete evaluation loop across the dataset and outputs formatted metrics.
        """
        total_episodes = min(len(dataloader), max_episodes) if max_episodes else len(dataloader)
        print(f"\n:=========== CD-FSS Clean Framework Evaluation: {benchmark_name.upper()} ===========")
        print(f"| Device:         {self.device}")
        print(f"| Adapter Type:   {self.adapter_type}")
        print(f"| Fusion Mode:    {self.fusion_mode} (temp={self.fusion_temp})")
        print(f"| Adapt Mode:     {self.adapt_mode}")
        print(f"| Total Episodes: {total_episodes}")
        print(f":========================================================================\n")

        dataset = getattr(dataloader, 'dataset', None)
        class_ids = getattr(dataset, 'class_ids', [0])
        tracker = MetricTracker(class_ids=class_ids)

        start_time = time.time()
        for idx, batch in enumerate(dataloader):
            if max_episodes is not None and idx >= max_episodes:
                break
            pred_mask, gt_mask, class_id = self.evaluate_episode(batch)
            tracker.update(pred_mask, gt_mask, class_id)

            if (idx + 1) % 50 == 0 or (idx + 1) == total_episodes or idx == 0:
                current_metrics = tracker.get_metrics()
                print(f"[Batch: {idx + 1:04d}/{total_episodes:04d}]  mIoU: {current_metrics['mIoU']:5.2f}%  |  FB-IoU: {current_metrics['FB-IoU']:5.2f}%")

        total_time = time.time() - start_time
        final_metrics = tracker.get_metrics()

        print(f"\n================ FINAL RESULTS ================")
        print(f"Benchmark:   {benchmark_name.upper()}")
        print(f"Adapter:     {self.adapter_type}")
        print(f"Fusion:      {self.fusion_mode}")
        print(f"Final mIoU:   {final_metrics['mIoU']:5.2f}%")
        print(f"Final FB-IoU: {final_metrics['FB-IoU']:5.2f}%")
        print(f"Elapsed Time: {total_time:.2f}s ({total_time / len(dataloader):.3f}s/episode)")
        print(f"================================================\n")

        return final_metrics
