# ENHANCED MEDICAL IMAGE PREPROCESSING FOR IRIS

import os
import glob
import nibabel as nib
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from scipy import ndimage
from skimage import exposure
import warnings

class EnhancedMedicalDataLoader:
    """
    Enhanced medical data loader with proper CT preprocessing for IRIS framework
    """

    def __init__(self, 
                 train_root_dir: str,
                 test_root_dir: str = None,
                 target_size: tuple = (128, 128, 128),
                 target_spacing: tuple = (1.5, 1.0, 1.0),  # mm spacing (z, y, x)
                 intensity_range: tuple = (-1000, 1000),
                 normalize_method: str = 'z_score',  # 'minmax', 'z_score', 'robust'
                 apply_clahe: bool = True,
                 gaussian_smooth: float = 0.5):
        """
        Args:
            target_spacing: Target voxel spacing in mm (depth, height, width)
            normalize_method: 'minmax', 'z_score', or 'robust'
            apply_clahe: Apply Contrast Limited Adaptive Histogram Equalization
            gaussian_smooth: Gaussian smoothing sigma (0 = no smoothing)
        """
        self.train_root_dir = Path(train_root_dir)
        self.test_root_dir = Path(test_root_dir) if test_root_dir else None
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.intensity_range = intensity_range
        self.normalize_method = normalize_method
        self.apply_clahe = apply_clahe
        self.gaussian_smooth = gaussian_smooth

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Scan for available cases
        self.train_cases = self._scan_cases(self.train_root_dir)
        self.test_cases = self._scan_cases(self.test_root_dir) if self.test_root_dir else []

        self.logger.info(f"Found {len(self.train_cases)} training cases")
        self.logger.info(f"Found {len(self.test_cases)} test cases")

        # Statistics for robust normalization
        self.global_stats = None

    def _scan_cases(self, root_dir: Path) -> List[str]:
        """Scan directory for AUTO_SEG_* cases"""
        if not root_dir.exists():
            self.logger.warning(f"Directory {root_dir} does not exist")
            return []

        cases = []
        for case_dir in root_dir.glob("AUTO_SEG_*"):
            if case_dir.is_dir():
                cases.append(case_dir.name)

        return sorted(cases)

    def compute_global_statistics(self, max_cases: int = 50):
        """
        Compute global dataset statistics for robust normalization
        """
        if not self.train_cases:
            return

        self.logger.info(f"Computing global statistics from {min(max_cases, len(self.train_cases))} cases...")
        
        all_intensities = []
        cases_to_sample = self.train_cases[:max_cases]
        
        for case_name in cases_to_sample:
            try:
                case_dir = self.train_root_dir / case_name
                image_path = case_dir / "data" / "img_fin.nii.gz"
                
                if not image_path.exists():
                    continue
                    
                image_nii = nib.load(str(image_path))
                image_data = image_nii.get_fdata().astype(np.float32)
                
                # Clip outliers and sample
                clipped = np.clip(image_data, self.intensity_range[0], self.intensity_range[1])
                sampled = clipped.flatten()[::100]  # Sample every 100th voxel for speed
                all_intensities.append(sampled)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {case_name} for statistics: {e}")
                continue

        if all_intensities:
            all_intensities = np.concatenate(all_intensities)
            self.global_stats = {
                'mean': np.mean(all_intensities),
                'std': np.std(all_intensities),
                'median': np.median(all_intensities),
                'q25': np.percentile(all_intensities, 25),
                'q75': np.percentile(all_intensities, 75),
                'q5': np.percentile(all_intensities, 5),
                'q95': np.percentile(all_intensities, 95)
            }
            
            self.logger.info(f"Global statistics computed:")
            self.logger.info(f"  Mean: {self.global_stats['mean']:.2f}")
            self.logger.info(f"  Std: {self.global_stats['std']:.2f}")
            self.logger.info(f"  Median: {self.global_stats['median']:.2f}")

    def load_case(self, root_dir: Path, case_name: str, 
                  image_type: str = "img_fin", 
                  mask_type: str = "gt_fin") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load and preprocess a single case with enhanced preprocessing
        """
        case_dir = root_dir / case_name

        # Paths to NIFTI files
        image_path = case_dir / "data" / f"{image_type}.nii.gz"
        mask_path = case_dir / "structure_set" / f"{mask_type}.nii.gz"

        # Check if files exist
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Load NIFTI files
        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))

        # Extract data and header info
        image_data = image_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.int64)
        
        # ADDED: Check for valid 3D shape before preprocessing
        if len(image_data.shape) != 3 or image_data.shape[0] < 3 or image_data.shape[1] < 3 or image_data.shape[2] < 3:
            self.logger.warning(f"Skipping {case_name}: Invalid 3D shape {image_data.shape}")
            return None
        
        # Get original spacing from header
        original_spacing = self._extract_spacing(image_nii)
        
        # Enhanced preprocessing pipeline
        image_tensor = self._enhanced_preprocess_image(image_data, original_spacing)
        mask_tensor = self._enhanced_preprocess_mask(mask_data, original_spacing)

        return image_tensor, mask_tensor

    def _extract_spacing(self, nii_image) -> tuple:
        """Extract voxel spacing from NIFTI header"""
        try:
            spacing = nii_image.header.get_zooms()
            # Return as (z, y, x) spacing
            if len(spacing) >= 3:
                return spacing[:3]
            else:
                return (1.0, 1.0, 1.0)
        except:
            self.logger.warning("Could not extract spacing from header, using default")
            return (1.0, 1.0, 1.0)

    def _enhanced_preprocess_image(self, image: np.ndarray, original_spacing: tuple) -> torch.Tensor:
        """
        Enhanced CT image preprocessing pipeline
        """
        # 1. Intensity clipping to remove extreme outliers
        image = np.clip(image, self.intensity_range[0], self.intensity_range[1])

        # 2. Gaussian smoothing for noise reduction (optional)
        if self.gaussian_smooth > 0:
            image = ndimage.gaussian_filter(image, sigma=self.gaussian_smooth)

        # 3. Resampling to target spacing
        image = self._resample_to_spacing(image, original_spacing, self.target_spacing)

        # 4. Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)

        # 5. Resize to target size
        if image_tensor.shape[1:] != self.target_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=self.target_size,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        # 6. Advanced normalization
        image_tensor = self._advanced_normalize(image_tensor)

        # 7. Contrast enhancement (CLAHE)
        if self.apply_clahe:
            image_tensor = self._apply_clahe_3d(image_tensor)

        return image_tensor

    def _enhanced_preprocess_mask(self, mask: np.ndarray, original_spacing: tuple) -> torch.Tensor:
        """
        Enhanced mask preprocessing with proper resampling
        """
        # 1. Resample to target spacing using nearest neighbor
        mask = self._resample_to_spacing(mask, original_spacing, self.target_spacing, order=0)

        # 2. Convert to tensor
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        # 3. Resize to target size using nearest neighbor
        if mask_tensor.shape != self.target_size:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=self.target_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()

        # 4. Clean up small disconnected components (optional)
        mask_tensor = self._clean_mask(mask_tensor)

        return mask_tensor

    def _resample_to_spacing(self, volume: np.ndarray, original_spacing: tuple, 
                            target_spacing: tuple, order: int = 1) -> np.ndarray:
        """
        Resample volume to target spacing using scipy
        """
        if original_spacing == target_spacing:
            return volume

        # Calculate zoom factors
        zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
        
        # Resample
        resampled = ndimage.zoom(volume, zoom_factors, order=order, prefilter=False)
        
        return resampled

    def _advanced_normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Advanced normalization methods
        """
        if self.normalize_method == 'minmax':
            # MinMax to [0, 1]
            min_val = self.intensity_range[0]
            max_val = self.intensity_range[1]
            image = (image - min_val) / (max_val - min_val)
            
        elif self.normalize_method == 'z_score':
            # Z-score normalization using global statistics or local statistics
            if self.global_stats is not None:
                mean = self.global_stats['mean']
                std = self.global_stats['std']
            else:
                # Use local statistics
                mean = image.mean()
                std = image.std()
            
            image = (image - mean) / (std + 1e-8)
            
        elif self.normalize_method == 'robust':
            # Robust normalization using percentiles
            if self.global_stats is not None:
                q5, q95 = self.global_stats['q5'], self.global_stats['q95']
                median = self.global_stats['median']
            else:
                q5, q95 = torch.quantile(image, torch.tensor([0.05, 0.95]))
                median = torch.median(image)
            
            # Clip to percentile range
            image = torch.clamp(image, q5, q95)
            # Normalize around median
            image = (image - median) / (q95 - q5 + 1e-8)
        
        return image

    def _apply_clahe_3d(self, image: torch.Tensor, clip_limit: float = 0.03) -> torch.Tensor:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram equalization) to 3D volume
        """
        try:
            # Convert to numpy for processing
            image_np = image.squeeze(0).numpy()  # Remove channel dimension
            
            # Apply CLAHE slice by slice (axial slices)
            enhanced_slices = []
            for i in range(image_np.shape[0]):
                slice_2d = image_np[i]
                
                # Normalize to [0, 1] for CLAHE
                slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
                
                # Apply CLAHE
                slice_enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit)
                enhanced_slices.append(slice_enhanced)
            
            # Stack and convert back to tensor
            enhanced_volume = np.stack(enhanced_slices, axis=0)
            return torch.from_numpy(enhanced_volume).unsqueeze(0)  # Add channel dimension back
            
        except Exception as e:
            self.logger.warning(f"CLAHE failed, using original image: {e}")
            return image

    def _clean_mask(self, mask: torch.Tensor, min_size: int = 100) -> torch.Tensor:
        """
        Clean mask by removing small disconnected components
        """
        try:
            mask_np = mask.numpy()
            cleaned_mask = np.zeros_like(mask_np)
            
            # Process each class separately
            unique_classes = np.unique(mask_np)
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                    
                # Binary mask for this class
                binary_mask = (mask_np == class_id).astype(np.uint8)
                
                # Label connected components
                labeled, num_features = ndimage.label(binary_mask)
                
                # Keep only large components
                for label_id in range(1, num_features + 1):
                    component_mask = (labeled == label_id)
                    if np.sum(component_mask) >= min_size:
                        cleaned_mask[component_mask] = class_id
            
            return torch.from_numpy(cleaned_mask.astype(np.int64))
        except:
            # Return original mask if cleaning fails
            return mask

    def load_train_data(self, 
                       image_type: str = "img_fin",
                       mask_type: str = "gt_fin",
                       max_cases: Optional[int] = None,
                       compute_stats: bool = True) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Load training data with enhanced preprocessing"""
        
        # Compute global statistics first if needed
        if compute_stats and self.global_stats is None and self.normalize_method in ['z_score', 'robust']:
            self.compute_global_statistics()
        
        images = []
        masks = []

        cases_to_load = self.train_cases[:max_cases] if max_cases else self.train_cases

        self.logger.info(f"Loading {len(cases_to_load)} training cases with enhanced preprocessing...")

        for i, case_name in enumerate(cases_to_load):
            try:
                # FIXED: Check if load_case returns a valid tuple
                result = self.load_case(self.train_root_dir, case_name, image_type, mask_type)
                if result is not None:
                    image, mask = result
                    images.append(image)
                    masks.append(mask)
                    
                    # Log detailed info for first few cases
                    if i < 3:
                        unique_labels = torch.unique(mask)
                        self.logger.info(f"Loaded {case_name}:")
                        self.logger.info(f"  Image: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
                        self.logger.info(f"  Mask: {mask.shape}, labels={unique_labels.tolist()}")
                        
            except Exception as e:
                self.logger.error(f"Failed to load {case_name}: {e}")
                continue

        self.logger.info(f"Successfully loaded {len(images)} cases")
        return images, masks

    def load_test_data(self,
                      image_type: str = "img_fin", 
                      mask_type: str = "gt_fin",
                      max_cases: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Load test data with same preprocessing as training"""
        if not self.test_root_dir:
            self.logger.warning("No test directory specified")
            return [], []

        images = []
        masks = []

        cases_to_load = self.test_cases[:max_cases] if max_cases else self.test_cases

        self.logger.info(f"Loading {len(cases_to_load)} test cases...")

        for case_name in cases_to_load:
            try:
                # FIXED: Check if load_case returns a valid tuple
                result = self.load_case(self.test_root_dir, case_name, image_type, mask_type)
                if result is not None:
                    image, mask = result
                    images.append(image)
                    masks.append(mask)
            except Exception as e:
                self.logger.error(f"Failed to load {case_name}: {e}")
                continue

        return images, masks

    def get_case_info(self) -> Dict:
        """Get comprehensive dataset information"""
        info = {
            'train_cases': self.train_cases,
            'test_cases': self.test_cases,
            'num_train': len(self.train_cases),
            'num_test': len(self.test_cases),
            'target_size': self.target_size,
            'target_spacing': self.target_spacing,
            'intensity_range': self.intensity_range,
            'normalization': self.normalize_method,
            'preprocessing': {
                'clahe': self.apply_clahe,
                'gaussian_smooth': self.gaussian_smooth,
                'spacing_resampling': True,
                'mask_cleaning': True
            }
        }
        
        if self.global_stats:
            info['global_statistics'] = self.global_stats
            
        return info

# INTEGRATION WITH IRIS TRAINING

def load_medical_datasets_enhanced(config: dict) -> tuple:
    """
    Enhanced medical dataset loading for IRIS training
    """
    logger = logging.getLogger(__name__)
    
    # Get data configuration
    data_config = config['data']
    
    # Initialize enhanced medical data loader
    loader = EnhancedMedicalDataLoader(
        train_root_dir="../AUTO_SEGMENTATION_TRAIN",
        test_root_dir="../AUTO_SEGMENTATION_TEST",
        target_size=tuple(data_config['input_size']),
        target_spacing=(1.5, 1.0, 1.0),  # Reasonable CT spacing
        intensity_range=tuple(data_config.get('clip_intensity', [-1000, 1000])),
        normalize_method='robust',  # Best for medical images
        apply_clahe=True,  # Enhance contrast
        gaussian_smooth=0.5  # Light smoothing
    )
    
    # Get dataset info
    info = loader.get_case_info()
    logger.info(f"Enhanced Dataset Info:")
    logger.info(f" Training cases: {info['num_train']}")
    logger.info(f" Test cases: {info['num_test']}")
    logger.info(f" Target size: {info['target_size']}")
    logger.info(f" Target spacing: {info['target_spacing']}")
    logger.info(f" Normalization: {info['normalization']}")
    
    if info['num_train'] == 0:
        logger.warning("No training cases found!")
        # Return empty lists to trigger fallback
        return [], [], [], []
    
    # Load training data with enhanced preprocessing
    max_train_cases = data_config.get('max_train_cases', None)
    train_images, train_masks = loader.load_train_data(
        image_type="img_fin",
        mask_type="gt_fin",
        max_cases=max_train_cases,
        compute_stats=True  # Compute global statistics
    )
    
    # Load validation data
    max_val_cases = data_config.get('max_val_cases', 5)
    val_images, val_masks = loader.load_test_data(
        image_type="img_fin",
        mask_type="gt_fin",
        max_cases=max_val_cases
    )
    
    # Use subset of training data for validation if no test data
    if len(val_images) == 0:
        logger.warning("No test cases found, using subset of training data")
        split_idx = max(1, len(train_images) // 5)
        val_images = train_images[-split_idx:]
        val_masks = train_masks[-split_idx:]
        train_images = train_images[:-split_idx]
        train_masks = train_masks[:-split_idx]
    
    logger.info(f"Final enhanced dataset:")
    logger.info(f" Training: {len(train_images)} cases")
    logger.info(f" Validation: {len(val_images)} cases")
    
    # Log sample statistics
    if len(train_images) > 0:
        sample_img = train_images[0]
        sample_mask = train_masks[0]
        logger.info(f" Sample image: {sample_img.shape}, range=[{sample_img.min():.3f}, {sample_img.max():.3f}]")
        logger.info(f" Sample mask: {sample_mask.shape}, classes={torch.unique(sample_mask).tolist()}")
    
    return train_images, train_masks, val_images, val_masks