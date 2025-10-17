# CORRECTED Training script with proper configuration handling
import os
import sys
import argparse
import yaml
import logging
import random
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
# Import Iris components - FIXED: Use consistent naming
from iris_model import IrisModel
from PIL import Image
from fixed_episodic_trainer import FixedEpisodicTrainer, FixedEpisodicDataset
from inference_engine import InferenceEngine
import torch.nn.functional as F
from data_loader import load_medical_datasets_enhanced as load_medical_datasets # Use the enhanced loader


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['log_level'].upper())
    
    # Create output directory
    output_dir = Path(config['experiment']['output_dir']) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Output directory: {output_dir}")
    
    return logger, output_dir
    
def create_model(config: dict) -> IrisModel:
    """Create Iris model from configuration"""
    model_config = config['model']
    
    model = IrisModel(
        in_channels=model_config['in_channels'],
        base_channels=model_config['base_channels'],
        embed_dim=model_config['embed_dim'],
        num_query_tokens=model_config['num_query_tokens'],
        num_classes=model_config['num_classes'],
        num_heads=model_config['num_heads'],
        num_blocks_per_stage=model_config['num_blocks_per_stage'],
        dropout=model_config['dropout'],
        deep_supervision=model_config['deep_supervision'],
        multi_scale=model_config['multi_scale']
    )
    
    return model

def create_datasets(config: dict, train_data: tuple, val_data: tuple) -> tuple:
    """Create episodic datasets for training - FIXED with augmentation"""
    train_images, train_masks = train_data
    val_images, val_masks = val_data
    data_config = config['data']
    
    # FIXED: Pass augmentation config for runtime augmentation
    train_dataset = FixedEpisodicDataset(
        images=train_images,
        masks=train_masks,
        n_way=config['data']['n_way'],      # Add this
        k_shot=config['data']['k_shot'],    # Add this  
        q_query=config['data']['q_query'],
        episode_length=data_config['episode_length'],
        augmentation_config=config.get('augmentation', {})  # ADD: Pass augmentation
    )
    
    # No augmentation for validation (important for fair evaluation)
    val_dataset = FixedEpisodicDataset(
        images=val_images,
        masks=val_masks,
        n_way=2,
        k_shot=1,
        q_query=1,
        episode_length=data_config['episode_length'] // 4
        # No augmentation_config for validation
    )
    
    return train_dataset, val_dataset

def run_training(config: dict, logger: logging.Logger, output_dir: Path):
    """Main training function"""
    
    # Set device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading datasets...")
    train_images, train_masks, val_images, val_masks = load_medical_datasets(config)
    logger.info(f"Loaded {len(train_images)} training samples, {len(val_images)} validation samples")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(
        config, (train_images, train_masks), (val_images, val_masks)
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} episodes")
    logger.info(f"Validation dataset: {len(val_dataset)} episodes")
    
    # Create model
    logger.info("Creating Iris model...")
    model = create_model(config)
    model_info = model.get_model_info()
    
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer
    training_config = config['training']
    
    trainer = FixedEpisodicTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        device=device,
        warmup_iterations=training_config['warmup_iterations'],
        max_iterations=training_config['max_iterations'],
        eval_interval=config['evaluation']['eval_interval'],
        save_interval=config['logging']['save_interval']
    )
    
    # Setup tensorboard if enabled
    if config['logging']['tensorboard']:
        writer = SummaryWriter(output_dir / 'tensorboard')
        logger.info(f"Tensorboard logging enabled: {output_dir / 'tensorboard'}")
    else:
        writer = None
    
    # Resume from checkpoint if specified
    resume_from = config['experiment'].get('resume_from')
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Start training
    logger.info("Starting training...")
    try:
        train_metrics, val_metrics = trainer.train()
        
        # Log final results
        if val_metrics:
            final_val = val_metrics[-1][1]
            logger.info(f"Training completed! Final validation DICE: {final_val['dice']:.4f}")
        
        # Save training history
        history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        torch.save(history, output_dir / 'training_history.pth')
        logger.info(f"Training history saved to {output_dir / 'training_history.pth'}")
        
        return trainer
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(output_dir / 'interrupted_checkpoint.pth')
        return trainer
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

def calculate_class_dice(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate dice score per class dynamically, based on unique labels in the targets.
    
    Args:
        predictions: Model output tensor, typically [B, C, D, H, W] (logits/probabilities).
        targets: Ground truth mask tensor, typically [B, D, H, W] (integer labels).
    
    Returns:
        A dictionary of class_id to dice score (float).
    """
    dice_scores = {}
    
    # 1. Convert predictions to a single hard-labeled tensor [D, H, W]
    if predictions.dim() == 5 and predictions.size(1) > 1:
        # Multi-class prediction: Argmax over channels to get hard labels (0-10)
        predictions = torch.argmax(predictions.cpu().squeeze(0), dim=0).long()
    elif predictions.dim() == 5 and predictions.size(1) == 1:
        # Binary prediction (single channel probability): Binarize
        predictions = (predictions.cpu().squeeze(0).squeeze(0) > 0.5).long()
    else:
        # Assume it's already a single-channel label map [D, H, W]
        predictions = predictions.cpu().squeeze().long()
        
    # 2. Prepare target mask
    if targets.dim() >= 4:
        targets = targets.cpu().squeeze().long()

    # 3. Find unique classes present in the ground truth (excluding background 0)
    unique_classes = torch.unique(targets).tolist()
    if 0 in unique_classes:
        unique_classes.remove(0)
    
    if not unique_classes:
        return {'all_classes': 1.0}

    # 4. Calculate Dice for each present class
    for class_id in unique_classes:
        # Create binary masks for this specific class
        pred_class = (predictions == class_id).float()
        target_class = (targets == class_id).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2. * intersection) / (union + 1e-8)
        
        # Cast to float for clean logging
        dice_scores[f'class_{class_id}'] = dice.item()
        
    return dice_scores

def run_real_inference(model: IrisModel, config: dict, logger: logging.Logger):
    """Run inference on a single test set sample and provide detailed output."""
    logger.info("="*50)
    logger.info("Running Real Inference (Single Sample)")
    logger.info("="*50)
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    engine = InferenceEngine(model, device=device)
    
    _, _, test_images, test_masks = load_medical_datasets(config)
    
    if not test_images:
        logger.warning("No test images found. Skipping inference demo.")
        return

    idx = 0
    query_image = test_images[idx].unsqueeze(0).to(device)
    reference_image = test_images[0].unsqueeze(0).to(device)
    reference_mask = test_masks[0].unsqueeze(0).to(device)
    
    logger.info(f"Processing Sample {idx}:")
    logger.info(f" - Query Image Shape: {query_image.shape}")
    logger.info(f" - Reference Image Shape: {reference_image.shape}")
    logger.info(f" - Reference Mask Shape: {reference_mask.shape}")

    result = engine.one_shot_inference(query_image, reference_image, reference_mask, task_id=f"test_sample_{idx}")
    
    # Load GT mask once
    gt_mask = test_masks[idx].cpu().squeeze(0).long()
    
    # CRITICAL FIX: Pass the raw model output (logits/probs) to calculate_class_dice
    # The function handles the argmax/binarization internally.
    class_wise_dice = calculate_class_dice(result['predictions'], gt_mask)
    
    logger.info("-" * 25)
    logger.info("Inference Results:")
    logger.info(f" - Predicted Mask Shape: {result['predictions'].shape}")
    logger.info(f" - Task Embedding Shape: {result['task_embedding'].shape}")
    logger.info(f" - Dice Score: {class_wise_dice}")
    logger.info("-" * 25)
    
    return result

def save_slice_as_png(volume: torch.Tensor, path_prefix: str, cmap: str = 'gray'):
    """Save all 2D slices of a 3D tensor to PNG files."""
    volume_np = volume.cpu().numpy()
    D = volume_np.shape[0]
    
    # Normalize volume for consistent grayscale plotting
    mn, mx = volume_np.min(), volume_np.max()
    if mx - mn < 1e-6:
        normalized_volume = volume_np # Avoid division by zero, volume is uniform
    else:
        normalized_volume = (volume_np - mn) / (mx - mn)
    
    for z in range(D):
        try:
            plt.imsave(f"{path_prefix}_slice_{z:03d}.png", normalized_volume[z], cmap=cmap)
        except Exception as e:
            logger.error(f"Failed to save slice {z} at {path_prefix}: {e}")

def calculate_single_binary_dice(pred_binary: torch.Tensor, target_binary: torch.Tensor) -> float:
    """Calculates Dice score for two aligned binary masks (1, D, H, W)."""
    smooth = 1e-6
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def calculate_class_dice(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculates multi-class Dice score. Expects raw output [B, C, D, H, W] and target [B, D, H, W].
    NOTE: This is retained for historical debugging but is NOT the primary episodic metric.
    """
    dice_scores = {}
    
    if predictions.dim() == 5 and predictions.size(1) > 1:
        # Multi-class prediction: Argmax over channels to get hard labels (0-10)
        predictions = torch.argmax(predictions.cpu().squeeze(0), dim=0).long()
    elif predictions.dim() == 5 and predictions.size(1) == 1:
        # Binary prediction: Binarize
        predictions = (predictions.cpu().squeeze(0).squeeze(0) > 0.5).long()
    else:
        predictions = predictions.cpu().squeeze().long()
        
    if targets.dim() >= 4:
        targets = targets.cpu().squeeze().long()

    unique_classes = torch.unique(targets).tolist()
    if 0 in unique_classes: unique_classes.remove(0)
    
    if not unique_classes: return {'all_classes': 1.0}

    for class_id in unique_classes:
        pred_class = (predictions == class_id).float()
        target_class = (targets == class_id).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2. * intersection) / (union + 1e-8)
        dice_scores[f'class_{class_id}'] = dice.item()
        
    return dice_scores

# --- Core Inference Function with Debugging ---

def run_real_inference_multi(model: IrisModel, config: dict, logger: logging.Logger, num_samples: int = 5):
    """
    Deep debug version of episodic inference, saving all intermediate tensors and plots.
    """
    output_dir = Path(config['experiment']['output_dir']) / config['experiment']['name'] / "debug_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"STARTING DEEP DEBUG INFERENCE ({num_samples} samples)")
    logger.info("="*80)
    
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    engine = InferenceEngine(model, device=device)
    
    _, _, test_images, test_masks = load_medical_datasets(config)
    
    if not test_images:
        logger.warning("No test images found. Cannot run inference.")
        return []

    all_dice_scores = []
    
    # 1. Define the Fixed Task (Reference)
    ref_mask_sample_0 = test_masks[0].cpu().squeeze().long()
    available_ref_classes = torch.unique(ref_mask_sample_0).tolist()
    if 0 in available_ref_classes: available_ref_classes.remove(0)
        
    task_class_id = available_ref_classes[0] if available_ref_classes else None 
    
    if task_class_id is None:
         logger.warning("Reference mask has no foreground classes. Using general foreground (ID 999).")
         task_class_id = 999
         ref_mask_binary = (ref_mask_sample_0 > 0).float().unsqueeze(0)
    else:
        ref_mask_binary = (ref_mask_sample_0 == task_class_id).float().unsqueeze(0)
    
    reference_image = test_images[0].unsqueeze(0).to(device)
    reference_mask = ref_mask_binary.unsqueeze(0).to(device) # [1, 1, D, H, W]

    logger.info(f"[TASK] Reference Class ID: {task_class_id}")
    logger.info(f"[TASK] Ref Mask Shape (Binary): {reference_mask.shape}")
    
    # --- Start Loop ---
    for idx in range(min(num_samples, len(test_images))):
        sample_dir = output_dir / f'sample_{idx}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("-" * 80)
        logger.info(f"Processing Sample {idx+1}/{min(num_samples, len(test_images))}")
        
        query_image = test_images[idx].unsqueeze(0).to(device)
        gt_mask_full = test_masks[idx].cpu().squeeze().long()
        
        # Validation Prints
        logger.debug(f"[VALIDITY] Query Image: {query_image.shape}, Sum: {query_image.sum():.2f}")
        logger.debug(f"[VALIDITY] Query GT (Full) Classes: {torch.unique(gt_mask_full).tolist()}")
        if not (query_image.sum() > 0): logger.error("CRITICAL: Query image sum is zero/empty!")
        if not (gt_mask_full > 0).sum(): logger.error("CRITICAL: Query GT mask is all background!")

        # 2. Run episodic inference
        result = engine.one_shot_inference(query_image, reference_image, reference_mask, task_id=f"test_sample_{idx}")
        
        # Extract predictions
        pred_logits_or_probs = result['predictions'].cpu().squeeze(0) # Shape [C, D, H, W]
        
        # --- Data Preparation for Metrics & Visualization ---
        
        # 2.1. Isolate GT for the specific task (Binary mask for the target object)
        if task_class_id == 999:
            gt_mask_isolated = (gt_mask_full > 0).long()
        else:
            gt_mask_isolated = (gt_mask_full == task_class_id).long() 
        
        # 2.2. Convert Prediction to Hard Binary Foreground (Argmax != 0)
        if pred_logits_or_probs.dim() == 4 and pred_logits_or_probs.size(0) > 1:
            # Multi-class output: Argmax -> Binary Foreground
            predicted_labels = torch.argmax(pred_logits_or_probs, dim=0).long()
            pred_mask_binary_for_dice = (predicted_labels != 0).long()
            pred_label_map = predicted_labels # Full label map for vis
        else:
            # Binary output: simple threshold
            pred_mask_binary_for_dice = (pred_logits_or_probs.squeeze(0) > 0.5).long()
            pred_label_map = pred_mask_binary_for_dice
            
        # 3. Compute Dice Score (Binary comparison)
        intersection = (pred_mask_binary_for_dice.float() * gt_mask_isolated.float()).sum()
        union = pred_mask_binary_for_dice.float().sum() + gt_mask_isolated.float().sum()
        avg_dice = (2.0 * intersection) / (union + 1e-8)
        
        logger.info(f"[METRIC] Dice Score (Task {task_class_id} overlap): {avg_dice:.4f}")
        logger.debug(f"[METRIC] Intersection: {intersection.item()}, Union: {union.item()}")

        # Check if GT for this specific task is empty
        if gt_mask_isolated.sum() == 0:
            logger.warning(f"GT for task Class {task_class_id} is EMPTY in Query Sample {idx}. Dice result is unreliable.")
        
        all_dice_scores.append(avg_dice.item())
        
        # --- Deep Debugging File Saving ---

        # 4.1. Save Raw Tensors
        torch.save(result['predictions'].cpu(), sample_dir / "raw_predictions.pt")
        torch.save(result['task_embedding'].cpu(), sample_dir / "task_embedding.pt")
        
        # 4.2. Save All GT/Pred Slices/Channels
        D = query_image.shape[-3]
        
        # Input Image Slices
        img_3d = query_image.cpu().squeeze().float()
        save_slice_as_png(img_3d, str(sample_dir / "query_image"), cmap='gray')

        # Reference Mask Slices
        save_slice_as_png(ref_mask_binary.cpu().squeeze(), str(sample_dir / "ref_mask_binary"), cmap='gray')

        # Predicted Label Map Slices (Multi-class/Final)
        save_slice_as_png(pred_label_map, str(sample_dir / "pred_label_map"), cmap='viridis')

        # GT Mask (Isolated) Slices
        save_slice_as_png(gt_mask_isolated, str(sample_dir / "gt_isolated"), cmap='gray')
        
        # Predicted Channel Slices (Save all C channels of the model output)
        if pred_logits_or_probs.dim() == 4:
            for c in range(pred_logits_or_probs.size(0)):
                save_slice_as_png(pred_logits_or_probs[c].float(), str(sample_dir / f"pred_channel_{c}"), cmap='plasma')
                
        logger.info(f"[FILES] Debug files saved to: {sample_dir}")

        # --- Matplotlib Plotting ---
        try:
            mid_slice = query_image.shape[-1] // 2 
            
            query_img_disp = query_image.cpu().squeeze(0).squeeze(0)
            ref_mask_disp = ref_mask_binary.cpu().squeeze(0).squeeze(0)
            
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            
            # 1. Query Image
            axs[0].imshow(query_img_disp[:, :, mid_slice], cmap='gray')
            axs[0].set_title("Query Image")
            
            # 2. Reference Mask (Binary Task Mask)
            axs[1].imshow(ref_mask_disp[:, :, mid_slice], cmap='gray')
            axs[1].set_title(f"Ref Mask (Task: {task_class_id})")
            
            # 3. Ground Truth Mask (Isolated)
            axs[2].imshow(gt_mask_isolated[:, :, mid_slice], cmap='gray')
            axs[2].set_title(f"GT Mask (Isolated)")
            
            # 4. Predicted Mask (Binary Result)
            axs[3].imshow(pred_mask_binary_for_dice[:, :, mid_slice], cmap='gray')
            axs[3].set_title(f"Predicted Mask (Dice: {avg_dice:.4f})")
            
            plt.suptitle(f"Test Sample {idx} - Task: Segment Class {task_class_id}")
            plt.savefig(output_dir / f"plot_summary_{idx}.png")
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Visualization failed for sample {idx}: {e}")

    logger.info("="*80)
    if all_dice_scores:
        avg_dice = sum(all_dice_scores) / len(all_dice_scores)
        logger.info(f"FINAL AVG DICE over {len(all_dice_scores)} samples: {avg_dice:.4f}")
    else:
        logger.warning("No valid Dice scores calculated.")
    logger.info("="*80)

    return all_dice_scores


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Iris Framework')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Setup logging
    logger, output_dir = setup_logging(config)
    
    logger.info("="*50)
    logger.info("IRIS Framework Training")
    logger.info("Universal Medical Image Segmentation via In-Context Learning")
    logger.info("="*50)
    
    # Log configuration
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Output directory: {output_dir}")
    

    # Full training
    try:
        trainer = run_training(config, logger, output_dir)
        
        run_real_inference(trainer.model, config, logger)

        run_real_inference_multi(trainer.model, config, logger)

        logger.info("Training and demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to complete training: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
