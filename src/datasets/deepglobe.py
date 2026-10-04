import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DeepglobeDataset(Dataset):
    """Clean Deepglobe Satellite Remote Sensing Dataset loader with auto-detection."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test'):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath

        # Auto-detect nested Deepglobe directory
        if os.path.exists(self.base_path) and not (os.path.exists(os.path.join(self.base_path, 'urban')) or os.path.exists(os.path.join(self.base_path, 'rangeland'))):
            for root, dirs, files in os.walk(self.base_path):
                if 'urban' in dirs or 'rangeland' in dirs:
                    self.base_path = root
                    print(f"[*] Deepglobe auto-detected base_path at: {self.base_path}")
                    break

        self.categories = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']
        self.class_ids = list(range(len(self.categories)))
        self.img_metadata_classwise, self.num_images = self._build_metadata()

    def _build_metadata(self):
        metadata = {}
        total = 0
        for cat in self.categories:
            cat_dir = os.path.join(self.base_path, cat)
            if not os.path.exists(cat_dir):
                metadata[cat] = []
                continue
            mask_paths = sorted(glob.glob(os.path.join(cat_dir, 'masks', '*.png')))
            valid = []
            for mp in mask_paths:
                m = np.array(Image.open(mp).convert('L'))
                if np.count_nonzero(m >= 128) > 0:
                    valid.append(mp)
            metadata[cat] = valid
            total += len(valid)
        return metadata, total

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        cat_idx = idx % len(self.categories)
        cat_name = self.categories[cat_idx]

        candidates = self.img_metadata_classwise[cat_name]
        chosen = np.random.choice(candidates, 1 + self.shot, replace=False)
        q_mask_path, s_mask_paths = chosen[0], chosen[1:]

        def mask_to_img(mp):
            return mp.replace('masks', 'images').replace('.png', '.jpg')

        q_img = Image.open(mask_to_img(q_mask_path)).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

        s_imgs = [Image.open(mask_to_img(smp)).convert('RGB') for smp in s_mask_paths]
        s_masks = [torch.tensor(np.array(Image.open(smp).convert('L')) >= 128).float() for smp in s_mask_paths]

        q_img_t = self.transform(q_img)
        q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

        s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
        s_masks_t = F.interpolate(torch.stack(s_masks).unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

        return {
            'query_img': q_img_t,
            'query_mask': q_mask_t,
            'support_set': (s_imgs_t, s_masks_t),
            'class_id': torch.tensor(cat_idx)
        }
