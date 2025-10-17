# augmentations.py
import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Tuple, Dict

class MedicalDataAugmentation:
    """
    Medical image augmentation for 3D volumes
    Implements augmentations mentioned in IRIS paper
    """
    
    def __init__(self, config: Dict):
        aug_config = config.get('augmentation', {})
        self.random_crop = aug_config.get('random_crop', False)
        self.random_flip = aug_config.get('random_flip', False) 
        self.random_rotation = aug_config.get('random_rotation', False)
        self.intensity_shift = aug_config.get('intensity_shift', False)
        self.gaussian_noise = aug_config.get('gaussian_noise', 0.0)
        self.elastic_deformation = aug_config.get('elastic_deformation', False)
        
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to image and mask pair
        Args:
            image: (C, D, H, W) tensor, normalized [0,1]
            mask: (D, H, W) tensor with class indices
        """
        # Random horizontal flip
        if self.random_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[3])  # Flip width
            mask = torch.flip(mask, dims=[2])
            
        # Random vertical flip  
        if self.random_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])  # Flip height
            mask = torch.flip(mask, dims=[1])
            
        # Random axial flip
        if self.random_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[1])  # Flip depth
            mask = torch.flip(mask, dims=[0])
            
        # Random 90-degree rotation in axial plane
        if self.random_rotation and random.random() < 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            image = torch.rot90(image, k=k, dims=[2, 3])
            mask = torch.rot90(mask, k=k, dims=[1, 2])
            
        # Intensity shift (brightness)
        if self.intensity_shift and random.random() < 0.5:
            shift = (random.random() - 0.5) * 0.2  # ±0.1 shift
            image = image + shift
            image = torch.clamp(image, 0.0, 1.0)
            
        # Gaussian noise
        if self.gaussian_noise > 0 and random.random() < 0.5:
            noise = torch.randn_like(image) * self.gaussian_noise
            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)
            
        # Random crop (if enabled)
        if self.random_crop and random.random() < 0.5:
            image, mask = self._random_crop(image, mask)
            
        # Elastic deformation (simplified version)
        if self.elastic_deformation and random.random() < 0.3:
            image, mask = self._elastic_deform(image, mask)
            
        return image, mask
    
    def _random_crop(self, image: torch.Tensor, mask: torch.Tensor, 
                    crop_ratio: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random crop and resize back"""
        C, D, H, W = image.shape
        
        # Calculate crop dimensions
        crop_d = int(D * crop_ratio)
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        
        # Random crop location
        start_d = random.randint(0, D - crop_d)
        start_h = random.randint(0, H - crop_h)  
        start_w = random.randint(0, W - crop_w)
        
        # Crop
        image_crop = image[:, start_d:start_d+crop_d, 
                          start_h:start_h+crop_h, 
                          start_w:start_w+crop_w]
        mask_crop = mask[start_d:start_d+crop_d,
                        start_h:start_h+crop_h,
                        start_w:start_w+crop_w]
        
        # Resize back to original dimensions
        image_resized = F.interpolate(image_crop.unsqueeze(0), 
                                    size=(D, H, W), 
                                    mode='trilinear', 
                                    align_corners=False).squeeze(0)
        mask_resized = F.interpolate(mask_crop.unsqueeze(0).unsqueeze(0).float(),
                                   size=(D, H, W),
                                   mode='nearest').squeeze(0).squeeze(0).long()
        
        return image_resized, mask_resized
    
    def _elastic_deform(self, image: torch.Tensor, mask: torch.Tensor,
                       alpha: float = 10.0, sigma: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified elastic deformation"""
        # This is a basic implementation - for production use specialized libraries
        # like elasticdeform or MONAI transforms
        
        # For now, return unchanged (implement with proper elastic deform library)
        return image, mask

def apply_augmentation(images: list, masks: list, config: dict) -> Tuple[list, list]:
    """
    Apply augmentation to list of images and masks
    """
    if not config.get('augmentation', {}):
        return images, masks
        
    augmenter = MedicalDataAugmentation(config)
    
    augmented_images = []
    augmented_masks = []
    
    for image, mask in zip(images, masks):
        aug_img, aug_mask = augmenter(image, mask)
        augmented_images.append(aug_img)
        augmented_masks.append(aug_mask)
        
    return augmented_images, augmented_masks
