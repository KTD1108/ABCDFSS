import sys
import os
import json
import csv
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

SCRATCH_DIR = r"C:\Users\TUF DASH\.gemini\antigravity-ide\brain\fc7c06c9-9758-4fd0-b1a2-24d92bf688c4\scratch"
PROJECT_ROOT = r"d:\xulyanhv2\ABCDFSS"

if SCRATCH_DIR not in sys.path:
    sys.path.insert(0, SCRATCH_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import runner
from core.backbone import Backbone as OfficialBackbone
from utils import commonutils as utils
from eval.evaluation import Evaluator
from eval.logger import AverageMeter

from src.models.backbone import ResNetBackbone as CleanBackbone
from src.engine.pipeline import CDFSSEngine
from src.metrics.metrics import MetricTracker
from src.datasets.builder import build_dataloader

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]

def run_investigation():
    print("=" * 80)
    print("      DEEP AUDIT: EXPLAINING E0 DISCREPANCY (OFFICIAL VS CLEAN ENGINE)")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: BACKBONE NUMERICAL VERIFICATION
    # ---------------------------------------------------------
    print("\n--- STEP 1: BACKBONE FEATURE NUMERICAL EQUIVALENCE ---")
    official_bb = OfficialBackbone('resnet50').to('cpu')
    clean_bb = CleanBackbone().to('cpu')

    torch.manual_seed(42)
    dummy_img = torch.randn(1, 3, 400, 400)
    with torch.no_grad():
        off_feats = official_bb.extract_feats(dummy_img)
        cln_feats = clean_bb.extract_features(dummy_img)

    print(f"Number of extracted feature layers: Official={len(off_feats)}, Clean={len(cln_feats)}")
    assert len(off_feats) == len(cln_feats) == 16, "Layer count mismatch!"

    for l_idx in [0, 3, 7, 12, 15]:
        off_f = off_feats[l_idx]
        cln_f = cln_feats[l_idx]
        abs_diff = torch.abs(off_f - cln_f)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (torch.abs(off_f) + 1e-8)).mean().item()
        print(f"Layer {l_idx:02d} ({off_f.shape}): max_abs_diff={max_diff:.8f}, mean_abs_diff={mean_diff:.8f}, rel_diff={rel_diff:.8f}")

    # ---------------------------------------------------------
    # STEP 2: LOAD MANIFEST & VERIFY EPISODE FAIRNESS
    # ---------------------------------------------------------
    print("\n--- STEP 2: EPISODE MANIFEST FAIRNESS & CHECKSUMS ---")
    manifest_path = os.path.join(PROJECT_ROOT, "experiments", "episodes", "lung_seed42_20episodes.json")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)

    episodes = manifest_data['episodes']
    base_datapath = r"C:\Users\TUF DASH\.cache\kagglehub\datasets\heyoujue\lungsegmentation\versions\2"

    print(f"Total episodes in manifest: {len(episodes)}")
    episode_records = []
    for ep in episodes:
        eid = ep['episode_id']
        q_path = os.path.join(base_datapath, ep['query_img'])
        s_path = os.path.join(base_datapath, ep['support_imgs'][0])
        qm_path = os.path.join(base_datapath, ep['query_mask'])
        sm_path = os.path.join(base_datapath, ep['support_masks'][0])

        q_hash = sha256_file(q_path)
        s_hash = sha256_file(s_path)
        qm_hash = sha256_file(qm_path)
        sm_hash = sha256_file(sm_path)
        episode_records.append({
            'eid': eid,
            'q_file': os.path.basename(q_path),
            's_file': os.path.basename(s_path),
            'q_hash': q_hash,
            's_hash': s_hash,
            'qm_hash': qm_hash,
            'sm_hash': sm_hash
        })
        if eid < 3:
            print(f"Episode {eid:02d}: Q={os.path.basename(q_path)} (hash={q_hash}) | S={os.path.basename(s_path)} (hash={s_hash})")

    # ---------------------------------------------------------
    # STEP 3: RUN OFFICIAL AUTHOR PIPELINE ON THE 20 EPISODES
    # ---------------------------------------------------------
    print("\n--- STEP 3: RUNNING OFFICIAL AUTHOR PIPELINE ---")
    runner.args.benchmark = 'lung'
    runner.args.datapath = base_datapath
    runner.args.nshot = 1

    # Load via official dataloader seeded with 42
    utils.fix_randseed(42)
    official_dl = runner.makeDataloader()
    official_batches = [b for idx, b in enumerate(official_dl) if idx < 20]

    # Initialize author FeatureMaker
    config_orig = runner.makeConfig()
    utils.fix_randseed(2)
    author_feat_maker = runner.makeFeatureMaker(official_dl.dataset, config_orig, device='cpu', randseed=2)

    official_results = []
    for idx, b in enumerate(official_batches):
        sseval = runner.SingleSampleEval(b, author_feat_maker)
        sseval.forward()
        fg_iou = sseval.calc_metrics()
        iou_val = fg_iou.item() if torch.is_tensor(fg_iou) else fg_iou

        fused = sseval.logit_mask[0]
        th = sseval.thresh.item() if torch.is_tensor(sseval.thresh) else sseval.thresh
        pred = sseval.pred_mask[0]
        gt = b['query_mask'][0]

        pred_fg = pred.sum().item()
        gt_fg = gt.sum().item()
        inter = sseval.area_inter[1].item()
        union = sseval.area_union[1].item()

        official_results.append({
            'eid': idx,
            'thresh': th,
            'pred_mean': fused.mean().item(),
            'pred_min': fused.min().item(),
            'pred_max': fused.max().item(),
            'pred_fg': pred_fg,
            'gt_fg': gt_fg,
            'inter': inter,
            'union': union,
            'iou': iou_val,
            'pred_tensor': pred.clone()
        })
        print(f"[Official] Ep {idx+1:02d}: Thresh={th:.4f}, IoU={iou_val*100:5.2f}%")

    off_mean_iou = np.mean([r['iou'] for r in official_results]) * 100
    off_cum_inter = sum([r['inter'] for r in official_results])
    off_cum_union = sum([r['union'] for r in official_results])
    off_cum_iou = (off_cum_inter / off_cum_union) * 100

    print(f"Official Mean Episode-IoU: {off_mean_iou:.2f}%")
    print(f"Official Cumulative IoU:   {off_cum_iou:.2f}%")

    # ---------------------------------------------------------
    # STEP 4: RUN CLEAN ENGINE E0 ON THE SAME 20 EPISODES
    # ---------------------------------------------------------
    print("\n--- STEP 4: RUNNING CLEAN ENGINE E0 ---")
    clean_dl = build_dataloader('lung', base_datapath, shot=1, img_size=400, bsz=1, nworker=0, split='test', manifest_path=manifest_path)
    clean_batches = [b for idx, b in enumerate(clean_dl) if idx < 20]

    # Test A: Clean Engine with default unconstrained seed
    engine_clean = CDFSSEngine(
        adapter_type='conv1x1',
        fusion_mode='mean',
        adapt_mode='first-episode',
        num_epochs=25,
        lr=1e-2,
        l0=3,
        device='cpu'
    )

    clean_results = []
    for idx, b in enumerate(clean_batches):
        pred_mask, gt_mask, cid = engine_clean.evaluate_episode(b)
        
        pred = pred_mask[0].float()
        gt = gt_mask[0].float()
        pred_fg = pred.sum().item()
        gt_fg = gt.sum().item()
        inter = (pred.bool() & gt.bool()).sum().item()
        union = (pred.bool() | gt.bool()).sum().item()
        iou_val = inter / max(union, 1e-6)

        clean_results.append({
            'eid': idx,
            'pred_fg': pred_fg,
            'gt_fg': gt_fg,
            'inter': inter,
            'union': union,
            'iou': iou_val,
            'pred_tensor': pred.clone()
        })
        print(f"[Clean E0] Ep {idx+1:02d}: IoU={iou_val*100:5.2f}%")

    cln_mean_iou = np.mean([r['iou'] for r in clean_results]) * 100
    cln_cum_inter = sum([r['inter'] for r in clean_results])
    cln_cum_union = sum([r['union'] for r in clean_results])
    cln_cum_iou = (cln_cum_inter / cln_cum_union) * 100

    print(f"Clean Mean Episode-IoU: {cln_mean_iou:.2f}%")
    print(f"Clean Cumulative IoU:   {cln_cum_iou:.2f}%")

    # ---------------------------------------------------------
    # STEP 5: EPISODE-BY-EPISODE COMPARISON TABLE & CSV
    # ---------------------------------------------------------
    print("\n--- STEP 5: EPISODE-BY-EPISODE DIRECT COMPARISON ---")
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "audit", "e0_episode_comparison.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "query_file", "support_file",
            "official_thresh", "official_pred_fg", "clean_pred_fg", "gt_fg",
            "official_iou", "clean_iou", "iou_diff_pp", "exact_mask_match"
        ])
        for idx in range(20):
            off = official_results[idx]
            cln = clean_results[idx]
            ep_rec = episode_records[idx]
            diff = (cln['iou'] - off['iou']) * 100
            mask_match = torch.equal(off['pred_tensor'], cln['pred_tensor'])
            writer.writerow([
                idx, ep_rec['q_file'], ep_rec['s_file'],
                f"{off['thresh']:.4f}", int(off['pred_fg']), int(cln['pred_fg']), int(off['gt_fg']),
                f"{off['iou']*100:.2f}", f"{cln['iou']*100:.2f}", f"{diff:+.2f}", mask_match
            ])
            print(f"Ep {idx+1:02d}: Official={off['iou']*100:5.2f}% | Clean={cln['iou']*100:5.2f}% | Diff={diff:+5.2f}% | Mask Match={mask_match}")

    print(f"\n[OK] CSV exported to: {csv_path}")

    # ---------------------------------------------------------
    # STEP 6: ISOLATING THE EXACT SOURCE OF DIFFERENCE
    # ---------------------------------------------------------
    print("\n--- STEP 6: ISOLATING THE EXACT MECHANISM ---")
    # Let's test Clean Engine when seeded with randseed=2 before adaptation setup
    print("[Test 6.1] Clean Engine with exact author randseed(2) during adaptation initialization:")
    utils.fix_randseed(2)
    engine_seeded = CDFSSEngine(
        adapter_type='conv1x1',
        fusion_mode='mean',
        adapt_mode='first-episode',
        num_epochs=25,
        lr=1e-2,
        l0=3,
        device='cpu'
    )
    # Seed torch right before setting up augmentator
    torch.manual_seed(2)
    engine_seeded.augmentator.setup_transforms()

    seeded_ious = []
    for idx, b in enumerate(clean_batches):
        pred_m, gt_m, _ = engine_seeded.evaluate_episode(b)
        p = pred_m[0].float()
        g = gt_m[0].float()
        inter = (p.bool() & g.bool()).sum().item()
        union = (p.bool() | g.bool()).sum().item()
        seeded_ious.append(inter / max(union, 1e-6))
    
    seeded_mean_iou = np.mean(seeded_ious) * 100
    print(f"Seeded Clean Mean Episode-IoU: {seeded_mean_iou:.2f}% (vs Official={off_mean_iou:.2f}%)")

if __name__ == '__main__':
    run_investigation()
