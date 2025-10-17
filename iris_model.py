# CRITICAL FIX: Memory Bank Integration in Training

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
from encoder_3d import Encoder3D, EncoderWithSkipConnections
from task_encoding import FixedTaskEncodingModule
from decoder import FixedQueryBasedDecoder
from dice_loss import FixedCombinedLoss, FixedDeepSupervisionLoss
from memory import FixedMemoryBank

class IrisModel(nn.Module):
    """
    CRITICAL FIXES:
    1. Memory bank integration in both training and inference
    2. Proper EMA updates during training
    3. Consistent shape handling throughout
    """

    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 embed_dim: int = 512,
                 num_query_tokens: int = 10,
                 num_classes: int = 1,
                 num_heads: int = 8,
                 num_blocks_per_stage: int = 2,
                 dropout: float = 0.1,
                 deep_supervision: bool = True,
                 multi_scale: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.deep_supervision = deep_supervision
        self.multi_scale = multi_scale

        # Encoder
        if multi_scale:
            self.encoder = EncoderWithSkipConnections(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks_per_stage=num_blocks_per_stage,
                dropout_rate=dropout
            )
        else:
            self.encoder = Encoder3D(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks_per_stage=num_blocks_per_stage
            )

        # Get encoder channels
        encoder_channels = self.encoder.get_feature_channels() if hasattr(self.encoder, 'get_feature_channels') else \
            self.encoder.encoder.get_feature_channels()
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",encoder_channels)
        # FIXED: Task encoding with proper query token evolution
        self.task_encoder = FixedTaskEncodingModule(
            feature_channels=encoder_channels[-1],
            embed_dim=embed_dim,
            num_query_tokens=num_query_tokens,
            num_heads=num_heads,
            dropout=dropout
        )

        # FIXED: Decoder with true simultaneous bidirectional attention
        self.decoder = FixedQueryBasedDecoder(
            encoder_channels=encoder_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            deep_supervision=deep_supervision
        )

        # CRITICAL FIX: Memory bank for EMA-based class storage
        self.memory_bank = FixedMemoryBank(
            embed_dim=embed_dim,
            num_query_tokens=num_query_tokens,
            max_classes=1000,
            ema_momentum=0.999  # Paper uses high momentum α=0.999
        )

        # Loss function
        self.criterion = FixedCombinedLoss(dice_weight=1.0, ce_weight=1.0)
        if deep_supervision:
            self.criterion = FixedDeepSupervisionLoss(self.criterion)

        # Task embedding cache for inference
        self.task_embedding_cache = {}
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode_image(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Encode image through shared encoder"""
        if hasattr(self.encoder, 'encoder'):  # EncoderWithSkipConnections
            main_features, skip_features = self.encoder(image)
            return main_features
        else:  # Standard Encoder3D
            return self.encoder(image)

    def encode_task(self,
                   reference_image: torch.Tensor,
                   reference_mask: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Task encoding with proper query token evolution
        """
        # Extract features from reference image
        ref_features = self.encode_image(reference_image)
        
        # Use deepest features for task encoding
        deep_features = ref_features[-1]
        
        # print(f"[DEBUG] IrisModel.encode_task called. deep_features.shape={deep_features.shape}")

        task_embedding = self.task_encoder(deep_features, reference_mask)
        
        return task_embedding

    def _update_memory_bank_ema(self, class_id: str, task_embedding: torch.Tensor):
        """
        CRITICAL FIX: EMA update for memory bank as per paper
        """
        if not hasattr(self, 'memory_bank') or self.memory_bank is None:
            return
        
        # Detach from computation graph to avoid gradient issues
        with torch.no_grad():
            # Handle different tensor shapes properly
            if task_embedding.dim() == 3:  # (B, tokens, embed_dim)
                if task_embedding.size(0) == 1:
                    embedding_to_store = task_embedding.squeeze(0)  # (tokens, embed_dim)
                else:
                    embedding_to_store = task_embedding.mean(dim=0)  # Average across batch
            elif task_embedding.dim() == 2:  # Already (tokens, embed_dim)
                embedding_to_store = task_embedding
            else:
                raise ValueError(f"Unexpected task embedding shape: {task_embedding.shape}")
            
            # Store in memory bank
            self.memory_bank.store_or_update(class_id, embedding_to_store, confidence=1.0)


    def forward(self, 
            query_image: torch.Tensor, 
            reference_image: Optional[torch.Tensor] = None,
            reference_mask: Optional[torch.Tensor] = None, 
            task_embedding: Optional[torch.Tensor] = None,
            class_id: Optional[str] = None,
            update_memory: bool = True) -> torch.Tensor:
        """
        CORRECTED: Proper memory bank usage for both training and inference
        """
        # Extract query features
        query_features = self.encode_image(query_image)
        
        # CORRECTED: Always try to retrieve from memory bank first
        retrieved_embedding = None
        if class_id and hasattr(self, 'memory_bank'):
            retrieved_embedding = self.memory_bank.retrieve_and_update(class_id)
        
        if task_embedding is None:
            if reference_image is None or reference_mask is None:
                if retrieved_embedding is not None:
                    # Use retrieved embedding from memory bank
                    if retrieved_embedding.dim() == 2:  # (tokens, embed_dim)
                        retrieved_embedding = retrieved_embedding.unsqueeze(0)  # (1, tokens, embed_dim)
                    task_embedding = retrieved_embedding
                else:
                    raise ValueError("Either task_embedding or (reference_image, reference_mask) must be provided")
            else:
                task_embedding = self.encode_task(reference_image, reference_mask)
                
        # CORRECTED: Update memory bank AFTER forward pass for both training and inference
        if class_id and update_memory and hasattr(self, 'memory_bank'):
            # Always update memory bank (training or inference)
            self.memory_bank.store_or_update(
                class_id, 
                task_embedding.detach(),  # Detach to avoid gradient issues
                confidence=1.0 if self.training else 0.8  # Higher confidence during training
            )
        
        # Decode with task guidance
        segmentation = self.decoder(query_features, task_embedding)
        return segmentation
    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Complete loss computation with consistent shape handling
        
        Args:
            predictions: Model predictions - either single tensor or list (deep supervision)
            targets: Ground truth targets
            
        Returns:
            Computed loss value
        """
        # Handle shape standardization for targets
        if isinstance(predictions, list):
            # Deep supervision case - use the deep supervision loss
            if targets.dim() == 4:  # (B, D, H, W)
                targets_for_loss = targets.long()
            elif targets.dim() == 5 and targets.size(1) == 1:  # (B, 1, D, H, W)
                targets_for_loss = targets.squeeze(1).long()
            elif targets.dim() == 3:  # (D, H, W) - add batch dimension
                targets_for_loss = targets.unsqueeze(0).long()
            else:
                targets_for_loss = targets.long()
                
            # Use deep supervision loss
            return self.criterion(predictions, targets_for_loss)
            
        else:
            # Single prediction case
            if targets.dim() == 5:  # (B, 1, D, H, W)
                targets_for_loss = targets.squeeze(1).long()
            elif targets.dim() == 4:  # (B, D, H, W)  
                targets_for_loss = targets.long()
            elif targets.dim() == 3:  # (D, H, W) - add batch dimension
                targets_for_loss = targets.unsqueeze(0).long()
            else:
                targets_for_loss = targets.long()
            
            # Ensure predictions have correct shape for loss computation
            if predictions.dim() == 4:  # (B, D, H, W) - add channel dimension
                predictions_for_loss = predictions.unsqueeze(1)
            else:
                predictions_for_loss = predictions
                
            # Handle binary vs multi-class segmentation
            if self.num_classes == 1:
                # Binary segmentation - ensure targets are in [0, 1] range
                targets_for_loss = torch.clamp(targets_for_loss.float(), 0, 1)
                if predictions_for_loss.size(1) == 1:
                    # Single channel output - use as-is
                    pass
                else:
                    # Multi-channel output - take first channel for binary
                    predictions_for_loss = predictions_for_loss[:, :1]
            else:
                # Multi-class segmentation - ensure targets are class indices
                targets_for_loss = torch.clamp(targets_for_loss, 0, self.num_classes - 1)
            
            # Compute loss using the criterion
            return self.criterion(predictions_for_loss, targets_for_loss) 
                    
    def one_shot_inference(self,
                          query_image: torch.Tensor,
                          reference_image: torch.Tensor,
                          reference_mask: torch.Tensor,
                          apply_sigmoid: bool = True,
                          threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """One-shot inference with proper task encoding"""
        self.eval()
        with torch.no_grad():
            # FIXED: Use corrected task encoding
            task_embedding = self.encode_task(reference_image, reference_mask)
            
            # Segment query image
            logits = self.forward(query_image, task_embedding=task_embedding)
            
            # Handle deep supervision
            if isinstance(logits, list):
                logits = logits[-1]
            
            # Convert to probabilities
            if apply_sigmoid:
                probabilities = torch.sigmoid(logits)
                prediction = (probabilities > threshold).float()
            else:
                probabilities = F.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1, keepdim=True).float()
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'logits': logits,
                'task_embedding': task_embedding
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'deep_supervision': self.deep_supervision,
            'multi_scale': self.multi_scale,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }
