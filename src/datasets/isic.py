import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ISICDataset(Dataset):
    """Clean ISIC 2018 Skin Lesion Dataset loader with robust directory structure detection."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test', manifest_path: str = None):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath
        self.manifest_path = manifest_path

        # Auto-detect nested ISIC directory
        if os.path.exists(self.base_path) and not os.path.exists(os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input')):
            for root, dirs, files in os.walk(self.base_path):
                if 'ISIC2018_Task1-2_Training_Input' in dirs:
                    self.base_path = root
                    print(f"[*] ISIC auto-detected base_path at: {self.base_path}")
                    break

        self.categories = ['1', '2', '3']
        self.class_ids = [0, 1, 2]
        self.episodes = None
        if self.manifest_path and os.path.exists(self.manifest_path):
            import json
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                self.episodes = manifest_data.get('episodes', manifest_data)
                print(f"[*] ISICDataset loaded {len(self.episodes)} fixed episodes from manifest: {self.manifest_path}")
        else:
            self.img_metadata_classwise, self.num_images = self._build_metadata()

    def _build_metadata(self):
        metadata = {}
        total = 0
        for cat in self.categories:
            cat_dir = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth', cat)
            paths = sorted(glob.glob(os.path.join(cat_dir, '*.png')))
            valid = []
            for p in paths:
                mask = np.array(Image.open(p).convert('L'))
                if np.count_nonzero(mask >= 128) > 0:
                    valid.append(p)
            metadata[cat] = valid
            total += len(valid)
        return metadata, total

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

        # Derive image path from mask path
        input_dir = os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input')
        q_name = os.path.splitext(os.path.basename(q_mask_path))[0].replace('_segmentation', '') + '.jpg'
        q_img_path = os.path.join(input_dir, q_name)

        q_img = Image.open(q_img_path).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

        s_imgs = []
        s_masks = []
        for smp in s_mask_paths:
            s_name = os.path.splitext(os.path.basename(smp))[0].replace('_segmentation', '') + '.jpg'
            s_imgs.append(Image.open(os.path.join(input_dir, s_name)).convert('RGB'))
            s_masks.append(torch.tensor(np.array(Image.open(smp).convert('L')) >= 128).float())

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
