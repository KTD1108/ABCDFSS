import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DeepglobeDataset(Dataset):
    """
    Clean Deepglobe Satellite Remote Sensing Few-Shot Segmentation Dataset loader.
    Supports standard CD-FSS episodic structure (classes '1'..'6' under test/origin and test/groundtruth).
    """
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test', manifest_path: str = None):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath
        self.manifest_path = manifest_path

        self.categories = ['1', '2', '3', '4', '5', '6']
        self.class_ids = list(range(len(self.categories)))
        self.episodes = None

        # Auto-detect if categories '1'..'6' are in datapath or a nested subfolder
        if os.path.exists(self.base_path):
            found = False
            for root, dirs, _ in os.walk(self.base_path):
                if any(c in dirs for c in self.categories):
                    # Check if category has test or origin inside
                    for c in self.categories:
                        if os.path.exists(os.path.join(root, c)):
                            self.base_path = root
                            found = True
                            print(f"[*] Deepglobe auto-detected base_path at: {self.base_path}")
                            break
                if found:
                    break

        if self.manifest_path and os.path.exists(self.manifest_path):
            import json
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                self.episodes = manifest_data.get('episodes', manifest_data)
                print(f"[*] DeepglobeDataset loaded {len(self.episodes)} fixed episodes from manifest: {self.manifest_path}")
        else:
            self.img_metadata_classwise, self.num_images = self._build_metadata()

    def _build_metadata(self):
        metadata = {}
        total = 0

        for cat in self.categories:
            metadata[cat] = []
            cat_dir = os.path.join(self.base_path, cat)
            if not os.path.exists(cat_dir):
                continue

            # Candidate image locations:
            # 1. <cat>/test/origin/*.jpg
            # 2. <cat>/origin/*.jpg
            # 3. <cat>/images/*.jpg
            img_patterns = [
                os.path.join(cat_dir, 'test', 'origin', '*.jpg'),
                os.path.join(cat_dir, 'origin', '*.jpg'),
                os.path.join(cat_dir, 'images', '*.jpg'),
                os.path.join(cat_dir, '*.jpg')
            ]

            img_paths = []
            for pat in img_patterns:
                matched = sorted(glob.glob(pat))
                if matched:
                    img_paths = matched
                    break

            metadata[cat] = img_paths
            total += len(img_paths)

        return metadata, total

    def _to_mask_path(self, img_path: str) -> str:
        """Finds matching mask path for given query/support image."""
        if 'origin' in img_path:
            return img_path.replace('origin', 'groundtruth').replace('.jpg', '.png')
        elif 'images' in img_path:
            return img_path.replace('images', 'masks').replace('.jpg', '.png')
        else:
            return img_path.replace('.jpg', '.png')

    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        return self.num_images

    def __getitem__(self, idx):
        if self.episodes is not None:
            ep = self.episodes[idx]
            q_img_path = ep['query_img'] if os.path.isabs(ep['query_img']) else os.path.join(self.base_path, ep['query_img'])
            q_mask_path = ep['query_mask'] if os.path.isabs(ep['query_mask']) else os.path.join(self.base_path, ep['query_mask'])
            s_img_paths = [p if os.path.isabs(p) else os.path.join(self.base_path, p) for p in ep['support_imgs']]
            s_mask_paths = [p if os.path.isabs(p) else os.path.join(self.base_path, p) for p in ep['support_masks']]
            cat_idx = ep.get('class_id', 0)

            q_img = Image.open(q_img_path).convert('RGB')
            q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

            s_imgs = [Image.open(p).convert('RGB') for p in s_img_paths]
            s_masks = [torch.tensor(np.array(Image.open(p).convert('L')) >= 128).float() for p in s_mask_paths]

            q_img_t = self.transform(q_img)
            q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

            s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
            s_masks_t = F.interpolate(torch.stack(s_masks).unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

            return {
                'query_img': q_img_t,
                'query_mask': q_mask_t,
                'support_set': (s_imgs_t, s_masks_t),
                'class_id': torch.tensor(cat_idx),
                'episode_id': ep.get('episode_id', idx),
                'query_img_name': ep.get('query_img', ''),
                'support_img_names': ep.get('support_imgs', []),
                'category': ep.get('category', str(cat_idx))
            }

        cat_idx = idx % len(self.categories)
        cat_name = self.categories[cat_idx]

        candidates = self.img_metadata_classwise[cat_name]
        if len(candidates) < 1 + self.shot:
            # Fallback if class has few images
            chosen = np.random.choice(candidates, 1 + self.shot, replace=True)
        else:
            chosen = np.random.choice(candidates, 1 + self.shot, replace=False)

        q_img_path = chosen[0]
        s_img_paths = chosen[1:]

        q_mask_path = self._to_mask_path(q_img_path)
        s_mask_paths = [self._to_mask_path(p) for p in s_img_paths]

        # Load images
        q_img = Image.open(q_img_path).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

        s_imgs = [Image.open(p).convert('RGB') for p in s_img_paths]
        s_masks = [torch.tensor(np.array(Image.open(p).convert('L')) >= 128).float() for p in s_mask_paths]

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
