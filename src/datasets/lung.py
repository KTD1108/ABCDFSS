import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LungDataset(Dataset):
    """Clean Lung / Chest X-ray Dataset loader with auto-detection and episode manifest support."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test', manifest_path: str = None):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath
        self.manifest_path = manifest_path

        # Auto-detect nested Lung directory
        if os.path.exists(self.base_path) and not os.path.exists(os.path.join(self.base_path, 'CXR_png')):
            for root, dirs, files in os.walk(self.base_path):
                if 'CXR_png' in dirs:
                    self.base_path = root
                    print(f"[*] Lung auto-detected base_path at: {self.base_path}")
                    break

        self.class_ids = [0]
        self.episodes = None
        if self.manifest_path and os.path.exists(self.manifest_path):
            import json
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                self.episodes = manifest_data.get('episodes', manifest_data)
                print(f"[*] LungDataset loaded {len(self.episodes)} fixed episodes from manifest: {self.manifest_path}")
        else:
            self.img_paths, self.mask_paths = self._build_metadata()

    def _build_metadata(self):
        cxr_dir = os.path.join(self.base_path, 'CXR_png')
        mask_dir = os.path.join(self.base_path, 'masks')
        all_masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

        valid_imgs = []
        valid_masks = []
        for mp in all_masks:
            name = os.path.splitext(os.path.basename(mp))[0].replace('_mask', '')
            ip = os.path.join(cxr_dir, f"{name}.png")
            if os.path.exists(ip):
                valid_imgs.append(ip)
                valid_masks.append(mp)

        return valid_imgs, valid_masks

    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.episodes is not None:
            ep = self.episodes[idx]
            q_img_path = os.path.join(self.base_path, ep['query_img'])
            q_mask_path = os.path.join(self.base_path, ep['query_mask'])
            s_img_paths = [os.path.join(self.base_path, p) for p in ep['support_imgs']]
            s_mask_paths = [os.path.join(self.base_path, p) for p in ep['support_masks']]
            class_id = ep.get('class_id', 0)

            q_img = Image.open(q_img_path).convert('RGB')
            q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()
            s_imgs = [Image.open(p).convert('RGB') for p in s_img_paths]
            s_masks = [torch.tensor(np.array(Image.open(p).convert('L')) >= 128).float() for p in s_mask_paths]
        else:
            all_indices = list(range(len(self.img_paths)))
            chosen = np.random.choice(all_indices, 1 + self.shot, replace=False)
            q_idx, s_indices = chosen[0], chosen[1:]

            q_img = Image.open(self.img_paths[q_idx]).convert('RGB')
            q_mask = torch.tensor(np.array(Image.open(self.mask_paths[q_idx]).convert('L')) >= 128).float()

            s_imgs = [Image.open(self.img_paths[i]).convert('RGB') for i in s_indices]
            s_masks = [torch.tensor(np.array(Image.open(self.mask_paths[i]).convert('L')) >= 128).float() for i in s_indices]
            class_id = 0

        q_img_t = self.transform(q_img)
        q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

        s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
        s_masks_t = F.interpolate(torch.stack(s_masks).unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

        return {
            'query_img': q_img_t,
            'query_mask': q_mask_t,
            'support_set': (s_imgs_t, s_masks_t),
            'class_id': torch.tensor(class_id)
        }
