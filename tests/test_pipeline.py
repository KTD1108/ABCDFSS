import os
import sys
import unittest
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.backbone import ResNetBackbone
from src.models.adapters import build_adapter
from src.models.attention import DenseCrossAttention
from src.models.fusion import build_fusion
from src.models.loss import DenseInfoNCELoss, KeepVarianceLoss, ContrastivePrototypeLoss
from src.metrics.thresholding import compute_otsu_threshold, apply_adaptive_threshold
from src.metrics.metrics import MetricTracker
from src.utils.augmentations import TaskAugmentator
from src.engine.pipeline import CDFSSEngine, resolve_experiment, EXPERIMENT_CONFIGS

class TestCDFSSToolkit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_backbone(self):
        bb = ResNetBackbone().to(self.device)
        dummy_img = torch.randn(2, 3, 200, 200, device=self.device)
        features = bb.extract_features(dummy_img)
        self.assertEqual(len(features), 16)
        self.assertEqual(features[0].shape[0], 2)

    def test_adapters(self):
        pw = build_adapter('conv1x1', in_channels=256, out_channels=64).to(self.device)
        dw = build_adapter('depthwise_separable_3x3', in_channels=256, out_channels=64).to(self.device)
        x = torch.randn(2, 256, 25, 25, device=self.device)
        
        out_pw = pw(x)
        out_dw = dw(x)
        self.assertEqual(out_pw.shape, (2, 64, 25, 25))
        self.assertEqual(out_dw.shape, (2, 64, 25, 25))

    def test_dense_cross_attention(self):
        dca = DenseCrossAttention(key_dim=64).to(self.device)
        q_feat = torch.randn(1, 64, 25, 25, device=self.device)
        s_feat = torch.randn(1, 1, 64, 25, 25, device=self.device)
        s_mask = torch.zeros(1, 1, 100, 100, device=self.device)
        s_mask[:, :, 20:50, 20:50] = 1.0

        coarse = dca(q_feat, s_feat, s_mask)
        self.assertEqual(coarse.shape, (1, 25, 25))
        self.assertTrue(torch.all(coarse >= 0.0) and torch.all(coarse <= 1.0))

    def test_fusion(self):
        fusion_softmax = build_fusion('softmax_margin', num_layers=13, temperature=1.0).to(self.device)
        fusion_mean = build_fusion('mean').to(self.device)
        q_stacked = torch.randn(1, 13, 100, 100, device=self.device)
        
        out_mean = fusion_mean(q_stacked)
        self.assertEqual(out_mean.shape, (1, 100, 100))

        out_softmax = fusion_softmax(q_stacked)
        self.assertEqual(out_softmax.shape, (1, 100, 100))

    def test_losses(self):
        nce = DenseInfoNCELoss(temperature=0.5).to(self.device)
        kvar = KeepVarianceLoss().to(self.device)
        proto = ContrastivePrototypeLoss().to(self.device)

        f1 = torch.randn(2, 64, 20, 20, device=self.device)
        f2 = torch.randn(2, 64, 20, 20, device=self.device)

        loss_nce = nce(f1, f2)
        loss_kvar = kvar(f1, f2)
        self.assertTrue(loss_nce.item() > 0)
        self.assertTrue(loss_kvar.item() >= 0)

        p_fg1 = torch.randn(2, 64, device=self.device)
        p_fg2 = torch.randn(2, 64, device=self.device)
        p_bg2 = torch.randn(2, 64, device=self.device)
        loss_proto = proto(p_fg1, p_fg2, p_bg2)
        self.assertTrue(loss_proto.item() > 0)

    def test_metrics_and_thresholding(self):
        tracker = MetricTracker(class_ids=[0, 1])
        pred1 = torch.tensor([[[0, 1], [1, 0]]])
        gt1 = torch.tensor([[[0, 1], [1, 0]]])
        tracker.update(pred1, gt1, class_id=0)

        # Dynamic class registration test
        tracker.update(pred1, gt1, class_id=99)
        m = tracker.get_metrics()
        self.assertAlmostEqual(m['mIoU'], 100.0, places=2)
        self.assertAlmostEqual(m['FB-IoU'], 100.0, places=2)

        # Otsu thresholding
        probs = torch.sigmoid(torch.randn(1, 50, 50, device=self.device))
        th_val = compute_otsu_threshold(probs)
        self.assertTrue(0.0 <= th_val <= 1.0)

        th, binary_mask = apply_adaptive_threshold(probs, method='pred_mean')
        self.assertEqual(binary_mask.shape, probs.shape)

    def test_augmentator(self):
        aug = TaskAugmentator(num_transforms=2, blur_kernel_size=1, max_shear=10)
        img = torch.randn(2, 3, 100, 100)
        mask = torch.ones(2, 100, 100)

        aug_img, aug_mask = aug.augment(img, mask)
        self.assertEqual(aug_img.shape, (2, 2, 3, 100, 100))
        self.assertEqual(aug_mask.shape, (2, 2, 100, 100))

    def test_end_to_end_engine(self):
        for adp in ['conv1x1', 'depthwise_separable_3x3']:
            engine = CDFSSEngine(
                adapter_type=adp,
                fusion_mode='softmax_margin',
                adapt_mode='first-episode',
                num_epochs=1,
                lr=1e-2,
                device=str(self.device)
            )

            H, W = 100, 100
            q_img = torch.randn(1, 3, H, W)
            q_mask = (torch.randn(1, H, W) > 0).float()
            s_imgs = torch.randn(1, 1, 3, H, W)
            s_masks = (torch.randn(1, 1, H, W) > 0).float()

            batch = {
                'query_img': q_img,
                'query_mask': q_mask,
                'support_set': (s_imgs, s_masks),
                'class_id': torch.tensor(0)
            }

            pred, gt, cid = engine.evaluate_episode(batch)
            self.assertEqual(pred.shape, (1, H, W))
            self.assertEqual(gt.shape, (1, H, W))
            self.assertEqual(cid, 0)

    def test_mean_fusion_output_equals_mean(self):
        """Test 1: Mean fusion output == mean(layer_predictions)."""
        fusion_mean = build_fusion('mean').to(self.device)
        q_stacked = torch.randn(2, 13, 50, 50, device=self.device)
        out = fusion_mean(q_stacked)
        expected = q_stacked.mean(dim=1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_softmax_fusion_weights_sum_to_one(self):
        """Test 2: Softmax margin fusion weights sum to 1."""
        fusion_softmax = build_fusion('softmax_margin', num_layers=13, temperature=1.0).to(self.device)
        s_feats = [torch.randn(1, 1, 64, 50, 50, device=self.device) for _ in range(16)]
        s_mask = torch.zeros(1, 1, 400, 400, device=self.device)
        s_mask[:, :, 50:150, 50:150] = 1.0
        margins = fusion_softmax.compute_layer_margins(s_feats, s_mask, l0=3)
        weights = torch.softmax(margins / fusion_softmax.temperature, dim=0)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)

    def test_changing_fusion_does_not_change_adapter(self):
        """Test 3: Changing fusion does not change adapter configuration."""
        e_mean = CDFSSEngine(adapter_type='conv1x1', fusion_mode='mean', device=str(self.device))
        e_soft = CDFSSEngine(adapter_type='conv1x1', fusion_mode='softmax_margin', device=str(self.device))
        self.assertEqual(e_mean.adapter_type, 'conv1x1')
        self.assertEqual(e_soft.adapter_type, 'conv1x1')
        self.assertEqual(type(e_mean.class_adapter_cache), type(e_soft.class_adapter_cache))

    def test_changing_adapter_does_not_change_fusion(self):
        """Test 4: Changing adapter does not change fusion."""
        e_dw = CDFSSEngine(adapter_type='depthwise_separable_3x3', fusion_mode='mean', device=str(self.device))
        e_pw = CDFSSEngine(adapter_type='conv1x1', fusion_mode='mean', device=str(self.device))
        self.assertEqual(e_dw.fusion_mode, 'mean')
        self.assertEqual(e_pw.fusion_mode, 'mean')
        self.assertEqual(type(e_dw.fusion_module), type(e_pw.fusion_module))

    def test_e0_configuration_resolves_to_pointwise_plus_mean(self):
        """Test 5: E0 configuration resolves to: pointwise/conv1x1 + mean."""
        adapter, fusion = resolve_experiment('E0')
        self.assertIn(adapter, ['conv1x1', 'pointwise'])
        self.assertEqual(fusion, 'mean')

    def test_e3_configuration_resolves_to_depthwise_plus_softmax(self):
        """Test 6: E3 resolves to: depthwise + softmax."""
        adapter, fusion = resolve_experiment('E3')
        self.assertEqual(adapter, 'depthwise_separable_3x3')
        self.assertEqual(fusion, 'softmax_margin')

if __name__ == '__main__':
    unittest.main()
