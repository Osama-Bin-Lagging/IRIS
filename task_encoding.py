import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from components import PixelShuffle3D, PixelUnshuffle3D

class FixedTaskEncodingModule(nn.Module):
    """
    FIXED Task encoding that produces T = [Tf, Tc] with consistent shape
    
    Key fixes:
    1. Tf is exactly 1 token (global foreground embedding)  
    2. Tc is exactly K tokens (learnable query tokens)
    3. Final shape is always [B, 1+K, embed_dim]
    4. Proper contextual encoding without PixelShuffle misuse
    5. CRITICAL FIX: Correct F.interpolate spatial dimension handling
    """
    
    def __init__(self, feature_channels: int, embed_dim: int = 512, num_query_tokens: int = 8, 
                 num_heads: int = 8, dropout: float = 0.1, downsample_factor: int = 2):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.embed_dim = embed_dim  
        self.num_query_tokens = num_query_tokens
        self.downsample_factor = downsample_factor
        
        # Calculate channels after PixelUnshuffle expansion
        r = downsample_factor
        expanded_channels = feature_channels * (r**3)  # PixelUnshuffle expands channels
        
        # Mask fusion after concatenation (features + mask channel)
        self.mask_fusion_conv = nn.Conv3d(
            expanded_channels + 1,  # +1 for mask channel
            embed_dim, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        
        # PixelShuffle operations for multi-resolution processing
        self.pixel_unshuffle = PixelUnshuffle3D(downsample_factor)
        self.pixel_shuffle = PixelShuffle3D(downsample_factor)
        
        # Proper normalization layers
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.norm_foreground = nn.LayerNorm(embed_dim)
        
        # Foreground feature projection
        self.foreground_proj = nn.Linear(feature_channels, embed_dim)
        
        # Learnable query tokens for contextual encoding
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Cross-attention for query token evolution
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network for query refinement
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_queries = nn.LayerNorm(embed_dim)

    def encode_foreground_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """CORRECTED Tf encoding with proper global pooling"""
        B, C, D, H, W = features.shape
        
        # CRITICAL FIX: Ensure mask has proper dimensions before interpolation
        if mask.shape[1:] != features.shape[2:]:  # Compare spatial dims only
            # Ensure mask is 5D for trilinear interpolation
            if mask.dim() == 4:  # (B, D, H, W)
                mask_for_interp = mask.unsqueeze(1)  # (B, 1, D, H, W) 
            else:
                mask_for_interp = mask
                
            mask_resized = F.interpolate(
                mask_for_interp.float(), 
                size=features.shape[2:],  # [D, H, W]
                mode='trilinear', 
                align_corners=False
            )
            # Remove channel dimension if it was added
            if mask.dim() == 4:
                mask_resized = mask_resized.squeeze(1)  # Back to (B, D, H, W)
        else:
            mask_resized = mask.float()
            
        # Ensure proper dimensions for broadcasting
        if mask_resized.dim() == 4:  # (B, D, H, W)
            mask_resized = mask_resized.unsqueeze(1)  # (B, 1, D, H, W)
            
        # Apply mask and compute global average
        masked_features = features * mask_resized
        mask_sum = mask_resized.sum(dim=(2, 3, 4), keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        # Global average pooling over masked region  
        valid_pixels = mask_resized.sum(dim=(2, 3, 4), keepdim=False)  # (B, 1)
        valid_pixels = torch.clamp(valid_pixels, min=1e-8)
        
        pooled = masked_features.sum(dim=(2, 3, 4)) / valid_pixels  # (B, C)
        
        # Project to embedding dimension
        Tf = self.foreground_proj(pooled).unsqueeze(1)  # (B, 1, embed_dim)
        Tf = self.norm_foreground(Tf)
        
        return Tf

    def encode_contextual_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """CORRECTED Proper PixelShuffle-based contextual encoding as per paper"""
        B, C, D, H, W = features.shape
        
        # CRITICAL FIX: Ensure mask has proper dimensions
        if mask.dim() == 4:  # (B, D, H, W)
            mask = mask.unsqueeze(1)  # (B, 1, D, H, W)
        
        # Resize mask to match features if needed
        if mask.shape[2:] != features.shape[2:]:
            mask = F.interpolate(
                mask.float(), 
                size=features.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        
        # Apply PixelUnshuffle to both features and mask
        features_unshuffled = self.pixel_unshuffle(features)  # (B, C*r^3, D/r, H/r, W/r)
        mask_unshuffled = self.pixel_unshuffle(mask)  # (B, 1*r^3, D/r, H/r, W/r)
        
        # Take only first channel of mask after unshuffle (they're all the same)
        mask_unshuffled = mask_unshuffled[:, :1]  # (B, 1, D/r, H/r, W/r)
        
        # Concatenate features and mask
        fused_input = torch.cat([features_unshuffled, mask_unshuffled], dim=1)
        
        # Apply mask fusion convolution
        contextual_features = self.mask_fusion_conv(fused_input)  # (B, embed_dim, D/r, H/r, W/r)
        
        # Flatten spatial dimensions for attention
        B, C, D_new, H_new, W_new = contextual_features.shape
        spatial_features = contextual_features.view(B, C, -1).transpose(1, 2)  # (B, D*H*W, embed_dim)
        spatial_features = self.norm_spatial(spatial_features)
        
        # Expand learnable query tokens for this batch
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, K, embed_dim)
        
        # Cross-attention: query tokens attend to spatial features
        contextual_queries, _ = self.cross_attention(query_tokens, spatial_features, spatial_features)
        
        # Apply residual connection and feed-forward network
        contextual_queries = query_tokens + contextual_queries
        contextual_queries = contextual_queries + self.ff_network(self.norm_queries(contextual_queries))
        
        return contextual_queries

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """CORRECTED Create T = [Tf, Tc] with proper PixelShuffle processing"""
        
        # Encode Tf (1 token)
        Tf = self.encode_foreground_features(features, mask)  # (B, 1, embed_dim)
        
        # Encode Tc (K tokens with PixelShuffle)  
        Tc = self.encode_contextual_features(features, mask)  # (B, K, embed_dim)
        
        # Concatenate to create final task embedding T = [Tf, Tc]
        task_embedding = torch.cat([Tf, Tc], dim=1)  # (B, 1+K, embed_dim)
        
        return task_embedding
        
    def get_output_shape(self, batch_size: int) -> tuple:
        """Get expected output shape"""
        return (batch_size, 1 + self.num_query_tokens, self.embed_dim)