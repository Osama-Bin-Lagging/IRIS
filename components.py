# Basic Building Components for Iris Framework

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

class ConvBlock3D(nn.Module):
    """3D Convolutional block with GroupNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))

class ResidualBlock3D(nn.Module):
    """3D Residual block for encoder"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock3D(out_channels, out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity)

class DownsamplingBlock3D(nn.Module):
    """3D Downsampling block with residual connections"""
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()
        layers = []
        
        # First block with stride=2 for downsampling
        layers.append(ResidualBlock3D(in_channels, out_channels, stride=2))
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for task encoding"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        return self.out_proj(out)

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for decoder"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query_features: torch.Tensor, task_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: query features attend to task embeddings
        query_attended = query_features + self.cross_attn(
            self.norm1(query_features), task_embeddings, task_embeddings
        )
        
        # Self-attention on task embeddings
        task_attended = task_embeddings + self.self_attn(
            self.norm2(task_embeddings), task_embeddings, task_embeddings
        )
        
        # Feed-forward
        query_out = query_attended + self.ff(self.norm3(query_attended))
        
        return query_out, task_attended

class PixelShuffle3D(nn.Module):
    """3D version of pixel shuffle for upsampling - FIXED channel calculation"""
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        r = self.upscale_factor
        
        # FIXED: Proper channel reduction calculation
        assert C % (r**3) == 0, f"Input channels {C} must be divisible by {r**3}"
        out_channels = C // (r**3)
        
        # Reshape and permute for upsampling
        x = x.view(B, out_channels, r, r, r, D, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, out_channels, D * r, H * r, W * r)
        return x

class PixelUnshuffle3D(nn.Module):
    """3D version of pixel unshuffle for downsampling - FIXED channel calculation"""
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        r = self.downscale_factor
        
        # FIXED: Ensure spatial dimensions are divisible
        assert D % r == 0 and H % r == 0 and W % r == 0, \
            f"Spatial dimensions must be divisible by {r}"
        
        # Reshape for downsampling
        x = x.view(B, C, D // r, r, H // r, r, W // r, r)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * (r**3), D // r, H // r, W // r)
        return x
