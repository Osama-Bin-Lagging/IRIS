# FIXED: True episodic training implementing N-way K-shot learning with Lamb optimizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging
from custom_optimizer import Lamb

class FixedEpisodicDataset(Dataset):
    """
    CORRECTED: True few-shot episodic dataset with proper class sampling
    """
    
    def __init__(self,
                 images: List[torch.Tensor],
                 masks: List[torch.Tensor],
                 n_way: int = 2,
                 k_shot: int = 1,  # CHANGED: Default k_shot to 1 for minimal requirements
                 q_query: int = 1,
                 episode_length: int = 1000,
                 augmentation_config: dict = None,
                 min_class_pixels: int = 50):  # ADDED: Minimum pixels for a class
        
        self.images = images
        self.masks = masks
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episode_length = episode_length
        self.min_class_pixels = min_class_pixels

        # Setup augmentation
        self.augmenter = None
        if augmentation_config:
            try:
                from augumentations import MedicalDataAugmentation
                self.augmenter = MedicalDataAugmentation({'augmentation': augmentation_config})
            except ImportError:
                pass

        # FIXED: Build proper class-to-samples mapping with lower requirements
        self.class_to_samples = self._build_class_sample_mapping()
        
        # Filter classes with sufficient examples
        min_examples = k_shot + q_query
        self.available_classes = [
            cls for cls, samples in self.class_to_samples.items()
            if len(samples) >= min_examples
        ]

        print(f"Found {len(self.class_to_samples)} total classes")
        print(f"Classes with enough examples ({min_examples}): {len(self.available_classes)}")
        
        # FALLBACK: If we don't have enough classes, reduce n_way or create synthetic classes
        if len(self.available_classes) < n_way:
            if len(self.available_classes) > 0:
                print(f"WARNING: Only {len(self.available_classes)} classes available, reducing n_way from {n_way}")
                self.n_way = min(len(self.available_classes), 2)  # At least try with 2-way
            else:
                # Last resort: create synthetic binary classes from any available masks
                print("WARNING: No proper classes found, creating synthetic binary tasks")
                self._create_synthetic_classes()

    def _build_class_sample_mapping(self):
        """
        FIXED: Build class-to-sample mapping with more flexible approach
        """
        class_to_samples = {}
        
        for sample_idx, mask in enumerate(self.masks):
            # Get unique anatomical classes in this mask
            unique_classes = torch.unique(mask)
            
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                
                # Create binary mask for this specific class
                binary_mask = (mask == class_id).float()
                
                # Only include if class has sufficient pixels
                if binary_mask.sum() >= self.min_class_pixels:
                    class_key = f"anatomy_{class_id.item()}"
                    if class_key not in class_to_samples:
                        class_to_samples[class_key] = []
                    
                    # Store image index and the specific binary mask for this class
                    class_to_samples[class_key].append({
                        'image_idx': sample_idx,
                        'binary_mask': binary_mask,
                        'class_id': class_id.item()
                    })
        
        return class_to_samples

    def _create_synthetic_classes(self):
        """
        Fallback: Create synthetic binary segmentation tasks from available masks
        """
        print("Creating synthetic classes from available masks...")
        self.class_to_samples = {}
        self.available_classes = []
        
        # For each mask, create multiple binary segmentation tasks by combining classes
        for sample_idx, mask in enumerate(self.masks):
            unique_classes = torch.unique(mask)
            non_zero_classes = [c for c in unique_classes if c > 0]
            
            if len(non_zero_classes) >= 1:
                # Create synthetic classes by grouping anatomical regions
                for i, class_id in enumerate(non_zero_classes):
                    # Create binary mask for this class vs all others
                    binary_mask = (mask == class_id).float()
                    
                    if binary_mask.sum() >= self.min_class_pixels:
                        synthetic_class_name = f"synthetic_binary_{sample_idx}_{i}"
                        
                        if synthetic_class_name not in self.class_to_samples:
                            self.class_to_samples[synthetic_class_name] = []
                        
                        self.class_to_samples[synthetic_class_name].append({
                            'image_idx': sample_idx,
                            'binary_mask': binary_mask,
                            'class_id': class_id.item()
                        })
        
        # Update available classes
        min_examples = self.k_shot + self.q_query
        self.available_classes = [
            cls for cls, samples in self.class_to_samples.items()
            if len(samples) >= min_examples
        ]
        
        # Adjust n_way if needed
        if len(self.available_classes) < self.n_way:
            self.n_way = max(1, len(self.available_classes))
        
        print(f"Created {len(self.available_classes)} synthetic classes")

    def __len__(self):
        return self.episode_length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        FIXED: Create proper N-way K-shot episode with fallback mechanisms
        """
        if len(self.available_classes) == 0:
            raise ValueError("No available classes for episodic training")
        
        # Sample N classes for this episode (with replacement if needed)
        if len(self.available_classes) >= self.n_way:
            episode_classes = random.sample(self.available_classes, self.n_way)
        else:
            # With replacement sampling if we don't have enough classes
            episode_classes = random.choices(self.available_classes, k=self.n_way)

        support_images = []
        support_masks = []
        query_images = []
        query_masks = []
        support_labels = []
        query_labels = []
        class_labels = []

        for class_idx, class_name in enumerate(episode_classes):
            class_samples = self.class_to_samples[class_name]
            
            # Ensure we have enough samples (with replacement if needed)
            required_samples = self.k_shot + self.q_query
            if len(class_samples) >= required_samples:
                selected_samples = random.sample(class_samples, required_samples)
            else:
                # Sample with replacement
                selected_samples = random.choices(class_samples, k=required_samples)

            # Split into support and query
            support_samples = selected_samples[:self.k_shot]
            query_samples = selected_samples[self.k_shot:]

            # Process support examples
            for sample_data in support_samples:
                image = self.images[sample_data['image_idx']].clone()
                mask = sample_data['binary_mask'].clone()

                # Apply augmentation if available
                if self.augmenter and random.random() < 0.7:
                    image, mask = self.augmenter(image, mask)

                support_images.append(image)
                support_masks.append(mask)
                support_labels.append(class_idx)
                class_labels.append(class_idx)

            # Process query examples
            for sample_data in query_samples:
                image = self.images[sample_data['image_idx']].clone()
                mask = sample_data['binary_mask'].clone()

                # Apply augmentation if available
                if self.augmenter and random.random() < 0.7:
                    image, mask = self.augmenter(image, mask)

                query_images.append(image)
                query_masks.append(mask)
                query_labels.append(class_idx)
                class_labels.append(class_idx)
        return {
            'support_images': torch.stack(support_images),  # (N*K, C, D, H, W)
            'support_masks': torch.stack(support_masks),    # (N*K, D, H, W)
            'support_labels': torch.tensor(support_labels),  # (N*K,)
            'query_images': torch.stack(query_images),      # (N*Q, C, D, H, W)
            'query_masks': torch.stack(query_masks),        # (N*Q, D, H, W)
            'query_labels': torch.tensor(query_labels),  # (N*Q,)
            'episode_classes': episode_classes,
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }

class FixedEpisodicTrainer:
    """
    FIXED: Episodic trainer with Lamb optimizer and additional parameters
    """
    
    def __init__(self,
                 model,
                 train_dataset: FixedEpisodicDataset,
                 val_dataset: Optional[FixedEpisodicDataset] = None,
                 learning_rate: float = 2e-3,  # Paper uses 2e-3 for Lamb
                 weight_decay: float = 1e-5,
                 device: str = 'cuda',
                 max_iterations: int = 50000,
                 eval_interval: int = 1000,
                 warmup_iterations: int = 1000,  # ADDED
                 save_interval: int = 5000):     # ADDED

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device(device)
        self.max_iterations = max_iterations
        self.eval_interval = eval_interval
        self.warmup_iterations = warmup_iterations  # ADDED
        self.save_interval = save_interval          # ADDED
        self.learning_rate = learning_rate
        # Move model to device
        self.model = self.model.to(self.device)

        # CRITICAL FIX: Setup Lamb optimizer as per paper
        self.optimizer = Lamb(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # Paper uses these values
            eps=1e-6,
            weight_decay=weight_decay,
            clamp_value=10.0  # Trust ratio clamp as in paper
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_iterations, eta_min=0.0)  # eta_min ensures lr reaches 0


        self.current_iteration = 0
        self.best_val_dice = 0.0
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using Lamb optimizer with lr={learning_rate}, wd={weight_decay}")
        self.logger.info(f"Warmup iterations: {warmup_iterations}")
        self.logger.info(f"Save interval: {save_interval}")

    def train_episode(self, episode: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        CORRECTED: Train on episode with proper memory bank integration and Dice metric calculation.
        """
        self.model.train()
        # Move episode to device
        support_images = episode['support_images'].to(self.device)
        support_masks = episode['support_masks'].to(self.device)
        support_labels = episode['support_labels'].to(self.device)
        query_images = episode['query_images'].to(self.device)
        query_masks = episode['query_masks'].to(self.device)
        query_labels = episode['query_labels'].to(self.device)
        n_way = episode['n_way']
        episode_classes = episode['episode_classes']
        total_loss = 0.0
        episode_dice = 0.0
        num_queries = 0
        self.optimizer.zero_grad()

        # Process each class in the episode
        for class_idx in range(n_way):
            support_mask = (support_labels == class_idx)
            query_mask = (query_labels == class_idx)
            
            if not support_mask.any() or not query_mask.any():
                continue
                
            # Get class data
            class_support_images = support_images[support_mask]
            class_support_masks = support_masks[support_mask]
            class_query_images = query_images[query_mask]
            class_query_masks = query_masks[query_mask]
            
            # Use first support example as reference
            ref_image = class_support_images[0:1]  # (1, C, D, H, W)
            ref_mask = class_support_masks[0:1]    # (1, D, H, W)
            if ref_mask.dim() == 3:
                ref_mask = ref_mask.unsqueeze(1)  # (1, 1, D, H, W)
            class_name = episode_classes[class_idx]
            
            # Process each query for this class
            for query_img, target_mask in zip(class_query_images, class_query_masks):
                query_img = query_img.unsqueeze(0)  # (1, C, D, H, W)
                
                # Forward pass
                predictions = self.model(
                    query_image=query_img,
                    reference_image=ref_image,
                    reference_mask=ref_mask,
                    class_id=class_name,
                    update_memory=True
                )
                
                # Compute loss
                if target_mask.dim() == 3:
                    target_for_loss = target_mask
                else:
                    target_for_loss = target_mask.squeeze()
                loss = self.model.compute_loss(predictions, target_for_loss)
                total_loss += loss
                num_queries += 1
                
                # Calculate DICE
                with torch.no_grad():
                    final_prediction = None # Reset for each query
                    
                    # FIX 1: Correctly define final_prediction
                    if predictions is not None:
                        if isinstance(predictions, list) and len(predictions) > 0:
                            final_prediction = predictions[-1]
                        elif not isinstance(predictions, list):
                            final_prediction = predictions
                    
                    if final_prediction is not None:
                        pred_probs = None
                        pred_binary = None

                        # Multi-class output logic (num_classes > 1, e.g., 11)
                        if final_prediction.size(1) > 1:
                            # Get the hard prediction mask (labels 0-10)
                            predicted_labels = torch.argmax(final_prediction, dim=1).long()
                            
                            # FIX 2: Define pred_binary as the foreground mask
                            # Predicted Foreground Mask (all non-background classes set to 1)
                            pred_binary = (predicted_labels != 0).float().unsqueeze(1)
                            
                            # Target is already guaranteed to be 0 or 1.
                            target_binary = target_mask.unsqueeze(0).unsqueeze(0).float()
                            
                        # Binary output logic (num_classes = 1 or default)
                        elif final_prediction.size(1) == 1:
                            pred_probs = torch.sigmoid(final_prediction)
                            pred_binary = (pred_probs > 0.5).float()
                            target_binary = target_mask.unsqueeze(0).unsqueeze(0).float()
                        
                        if pred_binary is not None:
                            intersection = (pred_binary * target_binary).sum()
                            dice = (2.0 * intersection + 1e-6) / (pred_binary.sum() + target_binary.sum() + 1e-6)
                            episode_dice += dice.item()
                        else:
                            episode_dice += 0.0
                    else:
                        episode_dice += 0.0

        # Backward pass
        if num_queries > 0:
            avg_loss = total_loss / num_queries
            avg_dice = episode_dice / num_queries
            
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # ADDED: Learning rate warmup
            if self.current_iteration < self.warmup_iterations:
                warmup_factor = self.current_iteration / self.warmup_iterations
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate * warmup_factor
            
            self.optimizer.step()
            self.scheduler.step()
            
            return {
                'loss': avg_loss.item(),
                'dice': avg_dice,
                'lr': self.optimizer.param_groups[0]['lr'],
                'num_queries': num_queries
            }
        else:
            return {'loss': 0.0, 'dice': 0.0, 'lr': self.optimizer.param_groups[0]['lr'], 'num_queries': 0}
        
    def train(self) -> Tuple[List, List]:
        """
        FIXED: Main training loop with proper episodic learning
        """
        self.logger.info("Starting fixed episodic training...")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Each "batch" is one episode
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: x[0]  # Extract single episode from batch
        )
        train_metrics = []
        val_metrics = []
        
        pbar = tqdm(total=self.max_iterations, desc="Training")
        
        for episode in train_loader:
            if self.current_iteration >= self.max_iterations:
                break
            # Train on episode
            metrics = self.train_episode(episode)
            # Log progress
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'DICE': f"{metrics['dice']:.4f}",
                'LR': f"{metrics['lr']:.6f}",
                'Queries': metrics['num_queries']
            })
            pbar.update(1)
            # Validation
            if (self.current_iteration % self.eval_interval == 0) and self.current_iteration != 0:
                if self.val_dataset is not None:
                    val_result = self.validate()
                    val_metrics.append((self.current_iteration, val_result))
                    if val_result['dice'] > self.best_val_dice:
                        self.best_val_dice = val_result['dice']
                        self.save_checkpoint('best_model.pth')
            # ADDED: Regular saving
            if self.current_iteration % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_iter_{self.current_iteration}.pth')
            train_metrics.append((self.current_iteration, metrics))
            self.current_iteration += 1
        pbar.close()
        return train_metrics, val_metrics
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """FIXED: Validation on episodic tasks"""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {'dice': 0.0, 'loss': 0.0}
        
        self.model.eval()
        
        def safe_collate(batch):
            if len(batch) == 0:
                raise StopIteration("Empty batch encountered")
            assert len(batch) == 1, f"Expected batch size 1, got {len(batch)}"
            return batch[0]
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=safe_collate
        )
        
        val_losses = []
        val_dice_scores = []
        
        for episode in val_loader:
            # Similar processing as training but without gradients
            support_images = episode['support_images'].to(self.device)
            support_masks = episode['support_masks'].to(self.device)
            support_labels = episode['support_labels'].to(self.device)
            query_images = episode['query_images'].to(self.device)
            query_masks = episode['query_masks'].to(self.device)
            query_labels = episode['query_labels'].to(self.device)
            
            episode_dice = 0.0
            episode_loss = 0.0
            num_queries = 0
            
            for class_idx in range(episode['n_way']):
                support_class_mask = (support_labels == class_idx)
                if not support_class_mask.any():
                    continue
                
                # Get support examples
                class_support_images = support_images[support_class_mask]
                class_support_masks = support_masks[support_class_mask]
                
                # Create task embedding
                primary_ref_image = class_support_images[0:1]
                primary_ref_mask = class_support_masks[0:1].unsqueeze(1)
                task_embedding = self.model.encode_task(primary_ref_image, primary_ref_mask)
                
                # Process queries
                query_class_mask = (query_labels == class_idx)
                if not query_class_mask.any():
                    continue
                
                class_query_images = query_images[query_class_mask]
                class_query_masks = query_masks[query_class_mask]
                
                for q_idx in range(class_query_images.size(0)):
                    query_img = class_query_images[q_idx:q_idx+1]
                    query_mask = class_query_masks[q_idx]
                    
                    # Forward pass
                    predictions = self.model(
                        query_image=query_img,
                        task_embedding=task_embedding
                    )
                    
                    # Compute metrics
                    loss = self.model.compute_loss(predictions, query_mask)
                    episode_loss += loss.item()
                    pred_probs = None
                    if isinstance(predictions, list):
                        pred_probs = torch.sigmoid(predictions[-1])
                    else:
                        pred_probs = torch.sigmoid(predictions)
                    
                    pred_binary = (pred_probs > 0.5).float()
                    target_binary = query_mask.unsqueeze(0).unsqueeze(0).float()
                    intersection = (pred_binary * target_binary).sum()
                    dice = (2.0 * intersection + 1e-6) / (
                        pred_binary.sum() + target_binary.sum() + 1e-6
                    )
                    episode_dice += dice.item()
                    num_queries += 1
            
            if num_queries > 0:
                val_losses.append(episode_loss / num_queries)
                val_dice_scores.append(episode_dice / num_queries)
        
        self.model.train()
        return {
            'loss': np.mean(val_losses) if val_losses else 0.0,
            'dice': np.mean(val_dice_scores) if val_dice_scores else 0.0
        }
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
        }
        torch.save(checkpoint, filename)
        self.logger.info(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.best_val_dice = checkpoint['best_val_dice']
        self.logger.info(f"Loaded checkpoint: {filename}")

    
def create_fixed_episodic_datasets(images: List[torch.Tensor],
                                 masks: List[torch.Tensor],
                                 train_split: float = 0.8,
                                 n_way: int = 2,
                                 k_shot: int = 1,  # CHANGED: Default to 1
                                 q_query: int = 1) -> Tuple[FixedEpisodicDataset, FixedEpisodicDataset]:
    """
    FIXED: Create proper train/val split for episodic learning
    """
    # Split data
    total_samples = len(images)
    train_size = int(total_samples * train_split)
    
    # Random split
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets
    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]

    train_dataset = FixedEpisodicDataset(
        train_images, train_masks,
        n_way=n_way, k_shot=k_shot, q_query=q_query,
        episode_length=1000
    )

    val_dataset = FixedEpisodicDataset(
        val_images, val_masks,
        n_way=n_way, k_shot=k_shot, q_query=q_query,
        episode_length=200  # Smaller validation set
    )

    return train_dataset, val_dataset