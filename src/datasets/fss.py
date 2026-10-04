import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FSS1000Dataset(Dataset):
    """Clean FSS-1000 Dataset loader with robust nested directory auto-detection."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test'):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath

        # Auto-detect nested fss1000 directory if extracted into subfolder
        if os.path.exists(self.base_path) and not os.path.exists(os.path.join(self.base_path, 'test.txt')):
            for root, dirs, files in os.walk(self.base_path):
                if 'test.txt' in files:
                    self.base_path = root
                    print(f"[*] FSS-1000 auto-detected base_path at: {self.base_path}")
                    break

        split_file = os.path.join(self.base_path, f'{split}.txt' if split in ['trn', 'val', 'test'] else 'test.txt')
        if not os.path.exists(split_file):
            # Fallback: scan class folders directly
            self.classes = sorted([d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))])
        else:
            with open(split_file, 'r') as f:
                self.classes = sorted([line.strip() for line in f.readlines() if line.strip()])

        self.class_ids = list(range(len(self.classes)))

    def __len__(self):
        return len(self.classes)

    def load_frame(self, query_img_path, query_mask_path, support_img_paths, support_mask_paths):
        q_img = Image.open(query_img_path).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(query_mask_path).convert('L'))).float()
        q_mask = (q_mask >= 128).float()

        s_imgs = [Image.open(p).convert('RGB') for p in support_img_paths]
        s_masks = []
        for p in support_mask_paths:
            m = torch.tensor(np.array(Image.open(p).convert('L'))).float()
            s_masks.append((m >= 128).float())

        return q_img, q_mask, s_imgs, torch.stack(s_masks)

    def __getitem__(self, idx):
        class_name = self.classes[idx % len(self.classes)]
        class_id = idx % len(self.classes)
        class_dir = os.path.join(self.base_path, class_name)

        # 10 images per class (1.jpg .. 10.jpg)
        all_ids = list(range(1, 11))
        chosen = np.random.choice(all_ids, 1 + self.shot, replace=False)
        q_id, s_ids = chosen[0], chosen[1:]

        q_img_path = os.path.join(class_dir, f"{q_id}.jpg")
        q_mask_path = os.path.join(class_dir, f"{q_id}.png")

        s_img_paths = [os.path.join(class_dir, f"{sid}.jpg") for sid in s_ids]
        s_mask_paths = [os.path.join(class_dir, f"{sid}.png") for sid in s_ids]

        q_img, q_mask, s_imgs, s_masks = self.load_frame(q_img_path, q_mask_path, s_img_paths, s_mask_paths)

        # Apply transforms
        q_img_t = self.transform(q_img)
        q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

        s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
        s_masks_t = F.interpolate(s_masks.unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

        return {
            'query_img': q_img_t,
            'query_mask': q_mask_t,
            'support_set': (s_imgs_t, s_masks_t),
            'class_id': torch.tensor(class_id)
        }
