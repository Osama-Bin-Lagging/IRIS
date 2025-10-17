import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FixedDiceLoss(nn.Module):
    """
    Simplified and robust Dice Loss for both binary and multi-class segmentation.
    """
    def __init__(self,
                 num_classes: int = 1,
                 smooth: float = 1e-6,
                 ignore_index: int = -100,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Standardize inputs to 5D: (B, C, D, H, W)
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
            
        if target.dim() == 3:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Apply activation based on the number of classes
        if self.num_classes > 1:
            pred_probs = F.softmax(pred, dim=1)
        else:
            pred_probs = torch.sigmoid(pred)
            
        # Handle deep supervision case where targets may be (B, 1, D, H, W)
        if target.size(1) == 1 and self.num_classes > 1:
            target = F.one_hot(target.squeeze(1).long(), num_classes=self.num_classes)
            target = target.permute(0, 4, 1, 2, 3).float()
        elif target.size(1) == 1 and self.num_classes == 1:
            target = target.float()

        # Reshape to (B, C, -1) for calculation
        pred_flat = pred_probs.view(pred_probs.size(0), pred_probs.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)

        # Compute loss
        dice_loss = 1.0 - dice_coefficient
        
        # Handle ignore_index (skip background class if multi-class)
        if self.num_classes > 1 and self.ignore_index == 0:
            dice_loss = dice_loss[:, 1:]

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

class FixedDeepSupervisionLoss(nn.Module):
    def __init__(self,
                 loss_fn: nn.Module,
                 weights: Optional[List[float]] = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights or [1.0, 0.8, 0.6, 0.4, 0.2]
        
    def forward(self, predictions: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not predictions or len(predictions) == 0:
            raise ValueError("Empty predictions list provided to DeepSupervisionLoss")
        
        total_loss = 0.0
        for i, pred in enumerate(predictions):
            if pred.shape[2:] != target.shape[-3:]:
                if target.dim() == 3:
                    target_for_resize = target.float().unsqueeze(0).unsqueeze(0)
                elif target.dim() == 4:
                    target_for_resize = target.float().unsqueeze(1)
                else:
                    target_for_resize = target.float()
                target_resized = F.interpolate(
                    target_for_resize,
                    size=pred.shape[2:],
                    mode='nearest'
                )
                if target.dim() == 3:
                    target_resized = target_resized.squeeze(0).squeeze(0).long()
                elif target.dim() == 4:
                    target_resized = target_resized.squeeze(1).long()
                else:
                    target_resized = target_resized.long()
            else:
                target_resized = target
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total_loss += weight * self.loss_fn(pred, target_resized)
        return total_loss

class FixedCombinedLoss(nn.Module):
    def __init__(self,
                 dice_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 smooth: float = 1e-6,
                 ignore_index: int = -100,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Determine num_classes for DiceLoss
        if class_weights is not None:
            num_classes = len(class_weights)
        else:
            # Assume num_classes based on common usage (e.g., binary)
            num_classes = 1
            
        self.dice_loss = FixedDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            ignore_index=ignore_index,
            weight=class_weights,
            reduction='mean'
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(pred, target)
        
        # Prepare target for CrossEntropyLoss
        if target.dim() == 5:
            target_for_ce = torch.argmax(target, dim=1).long()
        else:
            target_for_ce = target.long()

        # Handle binary vs multi-class CE
        num_classes = pred.size(1)
        if num_classes == 1:
            ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target_for_ce.float())
        else:
            ce_loss = self.ce_loss(pred, target_for_ce)

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss