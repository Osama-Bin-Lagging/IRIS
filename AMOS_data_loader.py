# ================================================================
# ENHANCED AMOS DATA LOADER (No Auto-Download, MONAI 1.5.1 Compatible)
# ================================================================

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from skimage import exposure
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ToTensord
)
from monai.data import Dataset


class EnhancedMedicalDataLoader:
    """
    Enhanced AMOS data loader (local only, no download).
    Loads AMOS data from a specified folder structure like:
        ./amos_data/imagesTr/*.nii.gz
        ./amos_data/labelsTr/*.nii.gz
    """

    def __init__(self,
                 train_root_dir: str,
                 test_root_dir: Optional[str] = None,
                 target_size: tuple = (128, 128, 128),
                 target_spacing: tuple = (1.5, 1.0, 1.0),
                 intensity_range: tuple = (-1000, 1000),
                 normalize_method: str = 'z_score',
                 apply_clahe: bool = True,
                 gaussian_smooth: float = 0.5):

        self.train_root_dir = Path(train_root_dir)
        self.test_root_dir = Path(test_root_dir) if test_root_dir else None
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.intensity_range = intensity_range
        self.normalize_method = normalize_method
        self.apply_clahe = apply_clahe
        self.gaussian_smooth = gaussian_smooth

        self.logger = logging.getLogger(__name__)

        # =========================================================
        # STEP 1 — VERIFY DATASET EXISTS
        # =========================================================
        if not self.train_root_dir.exists() or not any(self.train_root_dir.glob("imagesTr/*.nii.gz")):
            raise FileNotFoundError(
                f"❌ AMOS dataset not found in {self.train_root_dir}. "
                f"Please extract it manually to this path."
            )

        # =========================================================
        # STEP 2 — DEFINE BASE TRANSFORMS
        # =========================================================
        self.base_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"], a_min=intensity_range[0], a_max=intensity_range[1],
                b_min=0.0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image", "label"])
        ])

        # =========================================================
        # STEP 3 — BUILD DATASETS
        # =========================================================
        self.train_dataset = self._build_dataset(section="training")
        self.val_dataset = self._build_dataset(section="validation")

        self.test_dataset = None
        if self.test_root_dir and self.test_root_dir.exists():
            self.test_dataset = self._build_dataset(section="test", root=self.test_root_dir)

        self.logger.info(f"✅ AMOS Dataset initialized successfully:")
        self.logger.info(f"  Training cases: {len(self.train_dataset)}")
        self.logger.info(f"  Validation cases: {len(self.val_dataset)}")
        if self.test_dataset:
            self.logger.info(f"  Test cases: {len(self.test_dataset)}")

    # ================================================================
    # SUPPORTING FUNCTIONS
    # ================================================================

    def _build_dataset(self, section="training", root: Optional[Path] = None):
        """Construct MONAI dataset from AMOS folder structure."""
        root_dir = root or self.train_root_dir
        images_dir = root_dir / "imagesTr"
        labels_dir = root_dir / "labelsTr"

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"AMOS dataset folders not found at {root_dir}")

        all_images = sorted(images_dir.glob("*.nii.gz"))
        all_labels = sorted(labels_dir.glob("*.nii.gz"))

        if len(all_images) == 0 or len(all_labels) == 0:
            raise RuntimeError(f"No .nii.gz files found in {images_dir}")

        # simple split: 80% train, 20% val
        split_idx = int(0.8 * len(all_images))
        if section == "training":
            img_files = all_images[:split_idx]
            lbl_files = all_labels[:split_idx]
        elif section == "validation":
            img_files = all_images[split_idx:]
            lbl_files = all_labels[split_idx:]
        else:  # test
            img_files = all_images
            lbl_files = all_labels

        data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(img_files, lbl_files)]
        return Dataset(data=data_dicts, transform=self.base_transforms)

    # ================================================================
    # POST-PROCESSING FUNCTIONS
    # ================================================================

    def _apply_post_enhancements(self, image: torch.Tensor) -> torch.Tensor:
        img_np = image.squeeze(0).numpy()

        if self.gaussian_smooth > 0:
            img_np = ndimage.gaussian_filter(img_np, sigma=self.gaussian_smooth)

        if self.apply_clahe:
            enhanced_slices = []
            for i in range(img_np.shape[0]):
                slice_2d = img_np[i, :, :]
                slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                slice_enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=0.03)
                enhanced_slices.append(slice_enhanced)
            img_np = np.stack(enhanced_slices, axis=0)

        image = torch.from_numpy(img_np).unsqueeze(0).float()
        image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
        return image

    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.target_size, mode='nearest').squeeze(0).long()
        return mask

    # ================================================================
    # DATA LOADING FUNCTIONS
    # ================================================================

    def load_train_data(self, max_cases: Optional[int] = None):
        limit = max_cases if max_cases else len(self.train_dataset)
        images, masks = [], []
        for i in range(limit):
            data = self.train_dataset[i]
            image = self._apply_post_enhancements(data["image"])
            label = self._process_mask(data["label"])
            images.append(image)
            masks.append(label)
            if i < 3:
                self.logger.info(f"Train case {i}: Image {image.shape}, Mask {label.shape}")
        return images, masks

    def load_test_data(self, max_cases: Optional[int] = None):
        dataset = self.test_dataset or self.val_dataset
        if dataset is None:
            self.logger.warning("No test dataset found, returning empty list")
            return [], []
        limit = max_cases if max_cases else len(dataset)
        images, masks = [], []
        for i in range(limit):
            data = dataset[i]
            image = self._apply_post_enhancements(data["image"])
            label = self._process_mask(data["label"])
            images.append(image)
            masks.append(label)
        return images, masks

    def get_case_info(self) -> Dict:
        return {
            "train_cases": len(self.train_dataset),
            "val_cases": len(self.val_dataset),
            "test_cases": len(self.test_dataset) if self.test_dataset else 0,
            "target_size": self.target_size,
            "target_spacing": self.target_spacing,
            "intensity_range": self.intensity_range,
            "normalization": self.normalize_method,
            "clahe": self.apply_clahe,
            "gaussian_smooth": self.gaussian_smooth,
        }


# ================================================================
# INTEGRATION WRAPPER FOR IRIS
# ================================================================

def load_medical_datasets_enhanced(config: dict):
    logger = logging.getLogger(__name__)
    data_config = config["data"]

    loader = EnhancedMedicalDataLoader(
        train_root_dir='/users/student/rs/sandeep_k/IRIS/amos_data/amos22',
        test_root_dir=data_config.get("test_root", None),
        target_size=tuple(data_config["input_size"]),
        target_spacing=tuple(data_config.get("target_spacing", [1.5, 1.0, 1.0])),
        intensity_range=tuple(data_config.get("clip_intensity", [-1000, 1000])),
        normalize_method=data_config.get("normalization_method", "robust"),
        apply_clahe=data_config.get("apply_clahe", True),
        gaussian_smooth=data_config.get("gaussian_smooth", 0.5),
    )

    info = loader.get_case_info()
    logger.info(f"Loaded AMOS dataset with {info['train_cases']} training cases")

    train_images, train_masks = loader.load_train_data(max_cases=data_config.get("max_train_cases", None))
    val_images, val_masks = loader.load_test_data(max_cases=data_config.get("max_val_cases", 5))

    if len(val_images) == 0 and len(train_images) > 1:
        split_idx = max(1, len(train_images) // 5)
        val_images = train_images[-split_idx:]
        val_masks = train_masks[-split_idx:]
        train_images = train_images[:-split_idx]
        train_masks = train_masks[:-split_idx]

    logger.info(f"Training: {len(train_images)} | Validation: {len(val_images)}")
    return train_images, train_masks, val_images, val_masks
