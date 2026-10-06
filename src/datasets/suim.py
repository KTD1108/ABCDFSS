import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SUIMDataset(Dataset):
    """Clean SUIM Underwater Image Dataset loader with nested suim_merged auto-detection."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test', manifest_path: str = None):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath
        self.manifest_path = manifest_path

        # Auto-detect nested suim_merged directory
        if os.path.exists(os.path.join(self.base_path, 'suim_merged')):
            self.base_path = os.path.join(self.base_path, 'suim_merged')
            print(f"[*] SUIM auto-detected base_path at: {self.base_path}")
        elif os.path.exists(self.base_path) and not (os.path.exists(os.path.join(self.base_path, 'images')) and os.path.exists(os.path.join(self.base_path, 'masks'))):
            for root, dirs, files in os.walk(self.base_path):
                if 'images' in dirs and 'masks' in dirs:
                    self.base_path = root
                    print(f"[*] SUIM auto-detected base_path at: {self.base_path}")
                    break
                elif 'suim_merged' in dirs:
                    self.base_path = os.path.join(root, 'suim_merged')
                    print(f"[*] SUIM auto-detected base_path at: {self.base_path}")
                    break

        self.categories = ['FV', 'HD', 'PF', 'RI', 'RO', 'SR', 'WR']
        self.class_ids = list(range(len(self.categories)))
        self.img_path = os.path.join(self.base_path, 'images')
        self.ann_path = os.path.join(self.base_path, 'masks')
        self.episodes = None
        if self.manifest_path and os.path.exists(self.manifest_path):
            import json
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                self.episodes = manifest_data.get('episodes', manifest_data)
                print(f"[*] SUIMDataset loaded {len(self.episodes)} fixed episodes from manifest: {self.manifest_path}")
        else:
            self.img_metadata_classwise, self.num_images = self._build_metadata()

    def _build_metadata(self):
        metadata = {}
        total = 0
        for cat in self.categories:
            cat_masks = sorted(glob.glob(os.path.join(self.ann_path, cat, '*')))
            metadata[cat] = cat_masks
            total += len(cat_masks)
        return metadata, total

    def _resolve_image_path(self, maskpath: str) -> str:
        base_name = os.path.splitext(os.path.basename(maskpath))[0]
        jpg_p = os.path.join(self.img_path, base_name + '.jpg')
        if os.path.exists(jpg_p):
            return jpg_p
        png_p = os.path.join(self.img_path, base_name + '.png')
        if os.path.exists(png_p):
            return png_p
        return jpg_p

    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        return self.num_images

    def __getitem__(self, idx):
        if self.episodes is not None:
            ep = self.episodes[idx]
            q_img_rel = ep['query_img'].replace('\\', '/')
            q_mask_rel = ep['query_mask'].replace('\\', '/')
            q_img_path = q_img_rel if os.path.isabs(q_img_rel) else os.path.join(self.base_path, q_img_rel)
            q_mask_path = q_mask_rel if os.path.isabs(q_mask_rel) else os.path.join(self.base_path, q_mask_rel)
            s_img_paths = [p.replace('\\', '/') if os.path.isabs(p.replace('\\', '/')) else os.path.join(self.base_path, p.replace('\\', '/')) for p in ep['support_imgs']]
            s_mask_paths = [p.replace('\\', '/') if os.path.isabs(p.replace('\\', '/')) else os.path.join(self.base_path, p.replace('\\', '/')) for p in ep['support_masks']]
            class_id = ep.get('class_id', 0)

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
                'class_id': torch.tensor(class_id),
                'episode_id': ep.get('episode_id', idx),
                'query_img_name': ep.get('query_img', ''),
                'support_img_names': ep.get('support_imgs', []),
                'category': ep.get('category', str(class_id))
            }

        cat_idx = idx % len(self.categories)
        cat_name = self.categories[cat_idx]

        candidates = self.img_metadata_classwise[cat_name]
        chosen = np.random.choice(candidates, 1 + self.shot, replace=False)
        q_mask_path, s_mask_paths = chosen[0], chosen[1:]

        q_img = Image.open(self._resolve_image_path(q_mask_path)).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

        s_imgs = [Image.open(self._resolve_image_path(smp)).convert('RGB') for smp in s_mask_paths]
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
