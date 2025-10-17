# FIXED: Correct Bidirectional Cross-Attention implementing Equation 5

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CorrectBidirectionalCrossAttention(nn.Module):
    """
    FIXED: Implements Equation 5 from IRIS paper correctly
    F̂_q, T̂ = CrossAttention(F_q, T; T, F_q)
    
    This means TWO separate cross-attention operations:
    1. F_q attends to T (features query task embeddings)
    2. T attends to F_q (task embeddings query features)
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate attention mechanisms for each direction
        # Features attend to Task
        self.f2t_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.f2t_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.f2t_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.f2t_out = nn.Linear(embed_dim, embed_dim)
        
        # Task attends to Features  
        self.t2f_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t2f_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t2f_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t2f_out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_features = nn.LayerNorm(embed_dim)
        self.norm_task = nn.LayerNorm(embed_dim)
        
    def forward(self, query_features: torch.Tensor, task_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIXED: Implements Equation 5 with two separate cross-attention operations
        
        Args:
            query_features: F_q of shape (B, N_f, embed_dim)
            task_embeddings: T of shape (B, N_t, embed_dim)
            
        Returns:
            (F̂_q, T̂) - updated features and task embeddings
        """
        B, N_f, C = query_features.shape
        B, N_t, C = task_embeddings.shape
        
        # Normalize inputs
        features_norm = self.norm_features(query_features)
        task_norm = self.norm_task(task_embeddings)
        # Cross-Attention 1: Features attend to Task (F_q → T)
        q_f = self.f2t_query(features_norm).view(B, N_f, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.f2t_key(task_norm).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.f2t_value(task_norm).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_f2t = torch.matmul(q_f, k_t.transpose(-2, -1)) * self.scale
        attn_f2t = F.softmax(attn_f2t, dim=-1)
        out_f2t = torch.matmul(self.dropout(attn_f2t), v_t)
        out_f2t = out_f2t.transpose(1, 2).reshape(B, N_f, C)
        
        # Cross-Attention 2: Task attends to Features (T → F_q)
        q_t = self.t2f_query(task_norm).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        k_f = self.t2f_key(features_norm).view(B, N_f, self.num_heads, self.head_dim).transpose(1, 2)
        v_f = self.t2f_value(features_norm).view(B, N_f, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_t2f = torch.matmul(q_t, k_f.transpose(-2, -1)) * self.scale
        attn_t2f = F.softmax(attn_t2f, dim=-1)
        out_t2f = torch.matmul(self.dropout(attn_t2f), v_f)
        out_t2f = out_t2f.transpose(1, 2).reshape(B, N_t, C)
        
        # Apply residual connections and output projections
        updated_features = query_features + self.f2t_out(out_f2t)
        updated_task = task_embeddings + self.t2f_out(out_t2f)
        
        return updated_features, updated_task


class FixedQueryBasedDecoder(nn.Module):
    """
    FIXED: Decoder with correct bidirectional cross-attention
    """

    def __init__(self,
                 encoder_channels: list,
                 embed_dim: int = 512,
                 num_classes: int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 deep_supervision: bool = True):
        super().__init__()
        
        self.encoder_channels = encoder_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Feature projection layers
        self.feature_projections = nn.ModuleList([
            nn.Conv3d(ch, embed_dim, kernel_size=1) for ch in encoder_channels
        ])

        # FIXED: Correct bidirectional cross-attention
        self.bidirectional_attention = nn.ModuleList([
            CorrectBidirectionalCrossAttention(embed_dim, num_heads, dropout)
            for _ in encoder_channels
        ])

        # Upsampling layers for skip connections
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose3d(embed_dim, embed_dim, kernel_size=2, stride=2)
            for _ in range(len(encoder_channels) - 1)
        ])

        # Prediction heads
        if deep_supervision:
            self.prediction_heads = nn.ModuleList([
                nn.Conv3d(embed_dim, num_classes, kernel_size=1)
                for _ in encoder_channels
            ])
        else:
            self.prediction_heads = nn.ModuleList([
                nn.Conv3d(embed_dim, num_classes, kernel_size=1)
                 for _ in encoder_channels
            ])

        # Task aggregation for memory bank integration
        self.task_aggregation = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self,
                query_features: list,
                task_embedding: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Uses correct bidirectional cross-attention at each scale
        """
        predictions = []
        current_task_embedding = task_embedding
        
        # Start from deepest level and work upward
        x = None
        for i in reversed(range(len(query_features))):
            # Project features to embedding dimension
            feat = self.feature_projections[i](query_features[i])
            B, C, D, H, W = feat.shape
            
            # Flatten spatial dimensions for attention
            feat_flat = feat.view(B, C, -1).transpose(1, 2)  # (B, D*H*W, C)
            #print(f"[DEBUG] Decoder level {i}: feat_flat.shape = {feat_flat.shape}, current_task_embedding.shape = {current_task_embedding.shape}")

            # FIXED: Apply correct bidirectional cross-attention
            attended_features, current_task_embedding = self.bidirectional_attention[i](
                feat_flat, current_task_embedding
            )
            #print(f"[DEBUG] feat_flat.shape = {feat_flat.shape}, current_task_embedding.shape = {current_task_embedding.shape}")
            # Reshape back to spatial
            attended_features = attended_features.transpose(1, 2).view(B, C, D, H, W)
            
            # Skip connection from previous level
            if x is not None and i < len(self.upsample_layers):
                x_upsampled = self.upsample_layers[i](x)
                # Ensure spatial dimensions match
                if x_upsampled.shape[2:] != attended_features.shape[2:]:
                    x_upsampled = F.interpolate(
                        x_upsampled, size=attended_features.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                attended_features = attended_features + x_upsampled
                
            x = attended_features
            
            # Generate prediction at this scale
            if self.deep_supervision:
                pred = self.prediction_heads[i](x)
                predictions.append(pred)
        
        if self.deep_supervision:
            return predictions
        else:
            final_pred = self.prediction_heads[0](x)
            return final_pred