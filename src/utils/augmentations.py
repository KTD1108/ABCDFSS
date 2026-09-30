import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

class RandomAffineProxy:
    """Affine transformation proxy preserving spatial alignment across images and masks."""
    def __init__(self, max_angle: int = 0, max_shear: int = 20, max_scale: float = 1.0):
        self.angle = int(torch.randint(-max_angle, max_angle + 1, (1,)).item()) if max_angle > 0 else 0
        self.shear = [int(torch.randint(-max_shear, max_shear + 1, (1,)).item()) for _ in range(2)] if max_shear > 0 else [0, 0]
        self.scale = float(torch.rand(1).item() * (1.0 - max_scale) + max_scale) if max_scale < 1.0 else 1.0

    def apply(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return TF.affine(
            img_tensor,
            angle=self.angle,
            translate=[0, 0],
            scale=self.scale,
            shear=self.shear,
            interpolation=TF.InterpolationMode.BILINEAR
        )

    def apply_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        return TF.affine(
            mask_tensor,
            angle=self.angle,
            translate=[0, 0],
            scale=self.scale,
            shear=self.shear,
            interpolation=TF.InterpolationMode.NEAREST
        )


class TaskAugmentator:
    """
    Episode-level image and mask augmentator creating perturbed views
    for self-supervised test-time contrastive adaptation.
    """
    def __init__(self, num_transforms: int = 2, blur_kernel_size: int = 1, max_shear: int = 20):
        self.num_transforms = num_transforms
        self.blur_kernel_size = blur_kernel_size
        self.max_shear = max_shear

    def build_transforms(self):
        transforms = []
        for _ in range(self.num_transforms):
            blur = T.GaussianBlur(kernel_size=self.blur_kernel_size) if self.blur_kernel_size > 1 else (lambda x: x)
            affine = RandomAffineProxy(max_shear=self.max_shear)
            transforms.append((blur, affine))
        return transforms

    def augment(self, image: torch.Tensor, mask: torch.Tensor = None):
        """
        image: [B, 3, H, W]
        mask: [B, H, W] or [B, K, H, W]
        Returns:
            aug_images: [B, num_transforms, 3, H, W]
            aug_masks: [B, num_transforms, H, W] (if mask provided)
        """
        transforms = self.build_transforms()
        transformed_images = []
        transformed_masks = []

        for blur, affine in transforms:
            t_img = blur(image)
            t_img = affine.apply(t_img)
            transformed_images.append(t_img)

            if mask is not None:
                if mask.dim() == 4: # [B, K, H, W]
                    B, K, H, W = mask.shape
                    m_flat = mask.view(B * K, 1, H, W)
                    m_trans = affine.apply_mask(m_flat).view(B, K, H, W)
                    transformed_masks.append(m_trans)
                else:
                    t_mask = affine.apply_mask(mask.unsqueeze(1)).squeeze(1)
                    transformed_masks.append(t_mask)

        aug_images = torch.stack(transformed_images, dim=1)
        if mask is not None:
            aug_masks = torch.stack(transformed_masks, dim=1)
            return aug_images, aug_masks

        return aug_images
