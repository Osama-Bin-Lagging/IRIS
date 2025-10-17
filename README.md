# IRIS Implementation: Universal Medical Image Segmentation via In-Context Learning

This repository contains a comprehensive PyTorch implementation of the **IRIS (In-context Reference Image guided Segmentation)** framework from the paper "Show and Segment: Universal Medical Image Segmentation via In-Context Learning" (CVPR 2025).

## 🏥 Overview

IRIS is a novel framework that enables universal medical image segmentation through in-context learning, allowing adaptation to new segmentation tasks without fine-tuning by using reference image-label pairs.

### Key Features

- **Universal Framework**: Single model works across multiple anatomical structures and imaging modalities
- **In-Context Learning**: Adapts to new tasks using reference examples without retraining
- **Multiple Inference Strategies**: One-shot, ensemble, retrieval, and in-context tuning
- **3D Medical Imaging**: Native support for 3D medical volumes
- **Task Encoding**: Decoupled task definition from inference for efficiency

## 🏗️ Architecture

### Core Components

1. **3D UNet Encoder**: Multi-scale feature extraction with residual blocks
2. **Task Encoding Module**: Distills task-specific information from reference examples
   - Foreground feature encoding (high-resolution processing)
   - Contextual feature encoding (learnable query tokens with cross-attention)
3. **Query-Based Decoder**: Cross-attention mechanism for task-guided segmentation
4. **Multiple Inference Modes**: Flexible strategies for different scenarios

### Model Configuration

```yaml
# Default configuration (as per paper)
in_channels: 1                    # Grayscale medical images
base_channels: 32                 # Encoder progression: [32, 32, 64, 128, 256, 512]
embed_dim: 512                    # Task embedding dimension
num_query_tokens: 10              # Query tokens for contextual encoding
num_heads: 8                      # Multi-head attention
deep_supervision: true            # Multiple prediction scales
```

## 📦 Installation

### Requirements

```bash
# Create virtual environment
conda create -n iris python=3.8
conda activate iris

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- numpy, scipy, scikit-image
- nibabel, SimpleITK (for medical image loading)
- tensorboard, tqdm, pyyaml
- monai, albumentations (for data processing)

## 🚀 Quick Start

### 1. Basic Usage

```python
import torch
from iris_model import IrisModel
from inference_engine import InferenceEngine

# Initialize model
model = IrisModel(
    in_channels=1,
    base_channels=32,
    embed_dim=512,
    num_classes=1
)

# Create inference engine
engine = InferenceEngine(model, device='cuda')

# One-shot inference
result = engine.one_shot_inference(
    query_image=query_volume,      # (1, 1, D, H, W)
    reference_image=ref_volume,    # (1, 1, D, H, W)
    reference_mask=ref_mask,       # (1, 1, D, H, W)
    threshold=0.5
)

prediction = result['predictions']  # Binary segmentation mask
```

### 2. Training

```bash
# Train with default configuration
python train.py --config config.yaml

# Train with custom settings
python train.py --config custom_config.yaml --gpus 1

# Demo mode only (no training)
python train.py --demo-only
```

### 3. Different Inference Strategies

```python
# Context Ensemble (multiple references)
ensemble_result = engine.context_ensemble_inference(
    query_image=query,
    reference_images=[ref1, ref2, ref3],
    reference_masks=[mask1, mask2, mask3],
    ensemble_method='mean'
)

# Object-level Context Retrieval
retrieval_result = engine.object_level_context_retrieval(
    query_image=query,
    reference_pool=reference_database,
    k_retrievals=5,
    similarity_metric='cosine'
)

# In-context Tuning
tuned_result = engine.in_context_tuning(
    query_image=query,
    query_mask=ground_truth,      # For tuning
    reference_image=ref,
    reference_mask=ref_mask,
    tuning_steps=10
)
```

## 📊 Training Details

### Episodic Training

IRIS uses episodic training to simulate in-context learning scenarios:

```python
# Each training episode consists of:
# 1. Sample a class/task
# 2. Select support and query images from the same class
# 3. Encode task from support example
# 4. Predict segmentation for query image
# 5. Compute loss and update model

from episodic_trainer import EpisodicTrainer, EpisodicDataset

# Create episodic dataset
dataset = EpisodicDataset(
    images=train_images,
    masks=train_masks,
    episode_length=1000,
    max_classes_per_episode=3
)

# Initialize trainer
trainer = EpisodicTrainer(
    model=model,
    train_dataset=dataset,
    learning_rate=2e-3,
    max_iterations=80000,
    warmup_iterations=2000
)

# Start training
trainer.train()
```

### Training Configuration (config.yaml)

Key training parameters following the paper:

- **Max Iterations**: 80,000 (as per paper)
- **Learning Rate**: 2e-3 with warmup (2,000 iterations)
- **Optimizer**: AdamW with weight decay 1e-5
- **Loss**: Combined DICE + Cross-Entropy with deep supervision
- **Batch Size**: 4 (limited by 3D volume memory requirements)

### Loss Functions

```python
from dice_loss import CombinedLoss, DeepSupervisionLoss

# Combined DICE + Cross-Entropy Loss
criterion = CombinedLoss(dice_weight=1.0, ce_weight=1.0)

# Deep supervision for multi-scale training
deep_criterion = DeepSupervisionLoss(
    criterion, 
    weights=[1.0, 0.8, 0.6, 0.4, 0.2]
)
```

## 🔬 Medical Image Data Loading

### Example Data Loader (customize for your data)

```python
import nibabel as nib
import SimpleITK as sitk
import torch

def load_nifti_volume(image_path, mask_path):
    """Load NIFTI medical images and masks"""
    # Load image
    img = nib.load(image_path)
    image_data = torch.tensor(img.get_fdata()).float()
    
    # Load mask
    mask = nib.load(mask_path)
    mask_data = torch.tensor(mask.get_fdata()).long()
    
    # Add channel dimension
    image_data = image_data.unsqueeze(0)  # (1, D, H, W)
    
    return image_data, mask_data

# Preprocessing
def preprocess_volume(image, target_size=(128, 128, 128)):
    """Preprocess medical volume"""
    # Resample to target size
    image_resized = F.interpolate(
        image.unsqueeze(0), 
        size=target_size, 
        mode='trilinear',
        align_corners=False
    ).squeeze(0)
    
    # Intensity normalization
    image_norm = (image_resized - image_resized.mean()) / image_resized.std()
    
    return image_norm
```

### Supported Medical Image Formats

- **NIFTI** (.nii, .nii.gz) - Most common for research
- **DICOM** series - Clinical standard
- **MHA/MHD** - MetaImage format
- **NRRD** - Nearly Raw Raster Data

## 📈 Performance & Evaluation

### Metrics Implementation

```python
from dice_loss import dice_coefficient, iou_coefficient

# Calculate DICE coefficient
dice = dice_coefficient(predictions, ground_truth)

# Calculate IoU
iou = iou_coefficient(predictions, ground_truth)

# Hausdorff Distance (using SimpleITK)
import SimpleITK as sitk

hausdorff_filter = sitk.HausdorffDistanceImageFilter()
pred_sitk = sitk.GetImageFromArray(predictions.numpy())
gt_sitk = sitk.GetImageFromArray(ground_truth.numpy())
hausdorff_filter.Execute(pred_sitk, gt_sitk)
hausdorff_distance = hausdorff_filter.GetHausdorffDistance()
```

### Expected Performance

Based on the paper results:

- **In-distribution tasks**: Competitive with task-specific models
- **Out-of-distribution**: Superior generalization to new domains
- **Unseen classes**: Adapts through reference examples without retraining

## 🔧 Advanced Usage

### Custom Task Encoding

```python
# Encode custom task from multiple reference examples
task_embedding = model.task_encoder.encode_multiple_classes(
    features=encoder_features,
    masks=multi_class_masks  # (B, K, D, H, W)
)

# Cache task embeddings for efficiency
engine.task_embedding_cache['liver_segmentation'] = task_embedding
```

### Memory Bank for Seen Classes

```python
# Update memory bank with exponential moving average
engine.update_memory_bank('liver', task_embedding)

# Retrieve from memory bank
cached_embedding = engine.get_memory_bank_embedding('liver')
```

### Multi-Scale Processing

```python
# Enable multi-scale decoder
model = IrisModel(
    multi_scale=True,  # Enable multi-scale processing
    deep_supervision=True
)
```

## 📝 File Structure

```
iris-implementation/
├── components.py              # Basic building blocks
├── encoder_3d.py             # 3D UNet encoder
├── task_encoding.py          # Task encoding module
├── decoder.py                # Query-based decoder
├── dice_loss.py              # Loss functions
├── iris_model.py             # Main Iris model
├── episodic_trainer.py       # Training implementation
├── inference_engine.py       # Inference strategies
├── train.py                  # Training script
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🤝 SAM Integration

Since you mentioned providing SAM model code as reference, here's how to integrate:

```python
# Comparison with SAM-based approaches
from your_sam_model import SAMModel

# SAM requires interactive prompts
sam_model = SAMModel()
sam_result = sam_model.segment_with_prompts(
    image=query_image,
    point_prompts=click_points,
    box_prompts=bounding_boxes
)

# IRIS is fully automatic with reference examples
iris_result = engine.one_shot_inference(
    query_image=query_image,
    reference_image=reference_image,
    reference_mask=reference_mask
)

# IRIS advantages:
# - No manual prompts needed
# - Automatic processing
# - Better for high-throughput scenarios
# - Leverages task-specific context
```

## 🔬 Research Extensions

### Potential Improvements

1. **Multi-Modal Fusion**: Combine CT, MRI, PET modalities
2. **Temporal Segmentation**: 4D medical image sequences  
3. **Uncertainty Quantification**: Confidence measures for predictions
4. **Interactive Refinement**: User-guided improvements
5. **Edge Deployment**: Mobile/embedded medical devices

### Experimental Modifications

```python
# Add uncertainty quantification
class IrisWithUncertainty(IrisModel):
    def forward_with_uncertainty(self, *args, **kwargs):
        # Monte Carlo dropout for uncertainty
        predictions = []
        self.train()  # Enable dropout
        for _ in range(n_samples):
            pred = self.forward(*args, **kwargs)
            predictions.append(pred)
        
        mean_pred = torch.stack(predictions).mean(dim=0)
        uncertainty = torch.stack(predictions).var(dim=0)
        
        return mean_pred, uncertainty
```

## 📚 Citations

```bibtex
@inproceedings{gao2025iris,
    title={Show and Segment: Universal Medical Image Segmentation via In-Context Learning},
    author={Gao, Yunhe and Liu, Di and Li, Zhuowei and Li, Yunsheng and Chen, Dongdong and Zhou, Mu and Metaxas, Dimitris N.},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `base_channels` or `input_size` for smaller GPUs
2. **CUDA Out of Memory**: Use gradient checkpointing or reduce batch size
3. **Slow Training**: Enable mixed precision training
4. **Poor Convergence**: Check learning rate and warmup schedule

### Debug Mode

```python
# Enable debug logging
logging.getLogger().setLevel(logging.DEBUG)

# Test individual components
model.eval()
with torch.no_grad():
    # Test encoder
    features = model.encode_image(test_image)
    print(f"Encoder features: {[f.shape for f in features]}")
    
    # Test task encoding
    task_emb = model.encode_task(ref_image, ref_mask)
    print(f"Task embedding shape: {task_emb.shape}")
```

## 📄 License

This implementation is for research purposes. Please cite the original IRIS paper when using this code.

## 🤝 Contributing

Feel free to submit issues, improvements, or extensions to this implementation. Key areas for contribution:

- Additional medical image formats support
- More sophisticated data augmentation
- Integration with medical image databases
- Performance optimizations
- New inference strategies

---

**Note**: This implementation is based on the IRIS paper and provides a comprehensive framework for universal medical image segmentation. For production use, additional validation and testing on your specific medical imaging datasets is recommended.