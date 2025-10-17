# FIXED: Memory Bank with proper inference-time updates

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging
from collections import defaultdict
# FIXED: Memory Bank with proper inference-time updates and better structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import logging
from collections import defaultdict

class FixedMemoryBank(nn.Module):
    """
    CORRECTED: Memory Bank that updates during BOTH training and inference
    """

    def __init__(self,
                 embed_dim: int = 512,
                 num_query_tokens: int = 8,
                 max_classes: int = 1000,
                 ema_momentum: float = 0.999,
                 similarity_threshold: float = 0.85):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        self.task_embed_dim = (1 + num_query_tokens) * embed_dim  # Flattened T = [T_f; T_c]
        self.max_classes = max_classes
        self.ema_momentum = ema_momentum
        self.similarity_threshold = similarity_threshold
        self.access_counts = defaultdict(int)
        
        # Storage buffers
        self.register_buffer('class_embeddings', torch.zeros(max_classes, self.task_embed_dim))
        self.register_buffer('class_confidence', torch.zeros(max_classes))
        self.register_buffer('class_counts', torch.zeros(max_classes, dtype=torch.long))
        self.register_buffer('is_active', torch.zeros(max_classes, dtype=torch.bool))

        # Class mapping
        self.class_name_to_idx = {}
        self.idx_to_class_name = {}
        self.next_available_idx = 0

    def _get_or_create_class_idx(self, class_name: str) -> int:
        """Get or create index for class name"""
        if class_name not in self.class_name_to_idx:
            if self.next_available_idx >= self.max_classes:
                self._evict_least_confident()
            
            idx = self.next_available_idx
            self.class_name_to_idx[class_name] = idx
            self.idx_to_class_name[idx] = class_name
            self.next_available_idx += 1
            return idx
        
        return self.class_name_to_idx[class_name]

    def _evict_least_confident(self):
        """Evict least confident class when memory is full"""
        active_indices = torch.nonzero(self.is_active, as_tuple=True)[0]
        if len(active_indices) == 0:
            return

        min_confidence_idx = active_indices[torch.argmin(self.class_confidence[active_indices])]
        
        # Clear the slot
        self.class_embeddings[min_confidence_idx] = 0
        self.class_confidence[min_confidence_idx] = 0
        self.class_counts[min_confidence_idx] = 0
        self.is_active[min_confidence_idx] = False

        # Remove from mappings
        class_name = self.idx_to_class_name[min_confidence_idx.item()]
        del self.class_name_to_idx[class_name]
        del self.idx_to_class_name[min_confidence_idx.item()]
        
        self.next_available_idx = min_confidence_idx.item()

    def store_or_update(self,
                       class_name: str,
                       task_embedding: torch.Tensor,
                       confidence: float = 1.0,
                       force_update: bool = False) -> bool:
        """
        CORRECTED: Update during both training AND inference
        """
        with torch.no_grad():
            # Handle input shape standardization
            if task_embedding.dim() == 3:  # (B, 1+K, embed_dim)
                if task_embedding.size(0) == 1:
                    embedding_flat = task_embedding.squeeze(0).flatten()
                else:
                    embedding_flat = task_embedding.mean(dim=0).flatten()
            elif task_embedding.dim() == 2:  # (1+K, embed_dim)
                embedding_flat = task_embedding.flatten()
            else:  # Already flattened
                embedding_flat = task_embedding.clone()

            # Validate size
            if embedding_flat.size(0) != self.task_embed_dim:
                # Try to reshape if possible
                if embedding_flat.numel() == self.task_embed_dim:
                    embedding_flat = embedding_flat.view(-1)
                else:
                    return False

            # Normalize
            embedding_flat = F.normalize(embedding_flat, dim=0)
            
            idx = self._get_or_create_class_idx(class_name)
            
            if self.is_active[idx]:
                # CORRECTED: Always update (both training and inference)
                old_embedding = self.class_embeddings[idx]
                old_confidence = self.class_confidence[idx].item()
                
                # Confidence-weighted EMA
                effective_momentum = self.ema_momentum * (old_confidence / (old_confidence + confidence + 1e-8))
                if force_update:  # Lower momentum for forced updates during inference
                    effective_momentum = min(effective_momentum, 0.9)
                    
                new_embedding = effective_momentum * old_embedding + (1 - effective_momentum) * embedding_flat
                new_embedding = F.normalize(new_embedding, dim=0)
                
                self.class_embeddings[idx] = new_embedding
                self.class_confidence[idx] = max(old_confidence, confidence)
                self.class_counts[idx] += 1
            else:
                # Store new embedding
                self.class_embeddings[idx] = embedding_flat
                self.class_confidence[idx] = confidence
                self.class_counts[idx] = 1
                self.is_active[idx] = True
            
            return True
    def retrieve_and_update(self, class_name: str, 
                           new_embedding: Optional[torch.Tensor] = None,
                           confidence: float = 1.0) -> Optional[torch.Tensor]:
        """
        CORRECTED: Retrieve and optionally update (for inference-time adaptation)
        """
        if class_name not in self.class_name_to_idx:
            return None

        idx = self.class_name_to_idx[class_name]
        if not self.is_active[idx]:
            return None

        # Retrieve current embedding
        flat_embedding = self.class_embeddings[idx]
        reshaped = flat_embedding.view(1 + self.num_query_tokens, self.embed_dim)
        
        # Update with new embedding if provided (inference-time adaptation)
        if new_embedding is not None:
            self.store_or_update(class_name, new_embedding, confidence, force_update=True)
        
        return reshaped.clone()

    def retrieve(self, class_name: str) -> Optional[torch.Tensor]:
        """
        FIXED: Retrieve and reshape task embedding properly
        """
        if class_name not in self.class_name_to_idx:
            return None

        idx = self.class_name_to_idx[class_name]
        if not self.is_active[idx]:
            return None

        # Update access count
        self.access_counts[class_name] += 1

        # FIXED: Reshape back to (1+K, embed_dim) format
        flat_embedding = self.class_embeddings[idx]
        reshaped = flat_embedding.view(1 + self.num_query_tokens, self.embed_dim)
        
        return reshaped.clone()

    def retrieve_with_confidence(self, class_name: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve embedding with confidence score"""
        embedding = self.retrieve(class_name)
        if embedding is None:
            return None

        idx = self.class_name_to_idx[class_name]
        confidence = self.class_confidence[idx].item()
        
        return embedding, confidence

    def retrieve_similar(self, query_embedding: torch.Tensor, top_k: int = 3) -> List[Tuple[str, torch.Tensor, float]]:
        """
        ADDED: Retrieve similar embeddings based on cosine similarity
        """
        if not self.is_active.any():
            return []

        # Flatten query embedding if necessary
        if query_embedding.dim() > 1:
            query_flat = query_embedding.flatten()
        else:
            query_flat = query_embedding

        # Normalize query
        query_flat = F.normalize(query_flat, dim=0)

        # Get active embeddings
        active_indices = torch.nonzero(self.is_active, as_tuple=True)[0]
        active_embeddings = self.class_embeddings[active_indices]

        # Compute similarities
        similarities = F.cosine_similarity(query_flat.unsqueeze(0), active_embeddings, dim=1)

        # Get top-k
        top_k_actual = min(top_k, len(active_indices))
        top_similarities, top_indices = torch.topk(similarities, top_k_actual)

        results = []
        for i in range(top_k_actual):
            idx = active_indices[top_indices[i]].item()
            class_name = self.idx_to_class_name[idx]
            embedding = self.retrieve(class_name)
            similarity = top_similarities[i].item()
            results.append((class_name, embedding, similarity))

        return results

    def get_statistics(self) -> Dict:
        """Get comprehensive memory bank statistics"""
        active_count = self.is_active.sum().item()
        stats = {
            'active_classes': active_count,
            'total_capacity': self.max_classes,
            'utilization': active_count / self.max_classes if self.max_classes > 0 else 0,
            'total_accesses': sum(self.access_counts.values()),
            'class_names': list(self.class_name_to_idx.keys())
        }

        if active_count > 0:
            active_confidences = self.class_confidence[self.is_active]
            stats.update({
                'average_confidence': active_confidences.mean().item(),
                'min_confidence': active_confidences.min().item(),
                'max_confidence': active_confidences.max().item(),
                'average_count_per_class': self.class_counts[self.is_active].float().mean().item()
            })

        return stats

    def clear(self):
        """Clear all stored embeddings"""
        self.class_embeddings.zero_()
        self.class_confidence.zero_()
        self.class_counts.zero_()
        self.is_active.zero_()
        self.class_name_to_idx.clear()
        self.idx_to_class_name.clear()
        self.access_counts.clear()
        self.next_available_idx = 0
        
        self.logger.info("Memory bank cleared")

# Alias for backward compatibility
MemoryBank = FixedMemoryBank

class ObjectLevelContextRetrieval(nn.Module):
    """
    MISSING COMPONENT: Object-level Context Retrieval (IRIS Paper Section 3.4)
    
    This was only partially implemented in your inference_engine.py
    The paper describes a more sophisticated approach with object-level embeddings
    """
    
    def __init__(self, 
                 feature_extractor,
                 embed_dim: int = 512,
                 num_prototypes_per_class: int = 5):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.embed_dim = embed_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        
        # Prototype storage for each class
        self.memory_bank = FixedMemoryBank(embed_dim)
        
        # Learnable prototype aggregation
        self.prototype_aggregator = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Confidence scorer for retrieved prototypes
        self.confidence_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def extract_object_embeddings(self, 
                                image: torch.Tensor, 
                                mask: torch.Tensor) -> torch.Tensor:
        """
        Extract object-level embeddings using spatial attention
        
        This is more sophisticated than your simple pooling approach
        """
        # Extract features
        features = self.feature_extractor(image)
        deep_features = features[-1]  # Use deepest features
        
        # Spatial attention based on mask
        B, C, D, H, W = deep_features.shape
        mask_resized = F.interpolate(
            mask.float(), size=(D, H, W), 
            mode='trilinear', align_corners=False
        )
        
        # Weighted spatial attention
        attention_weights = mask_resized / (mask_resized.sum(dim=(2,3,4), keepdim=True) + 1e-8)
        
        # Extract object-centric features
        weighted_features = deep_features * attention_weights
        
        # Multiple pooling strategies for richer representation
        global_pool = F.adaptive_avg_pool3d(weighted_features, 1).flatten(1)
        max_pool = F.adaptive_max_pool3d(weighted_features, 1).flatten(1)
        
        # Combine different pooling strategies
        object_embedding = torch.cat([global_pool, max_pool], dim=1)
        
        # Project to embedding dimension
        if object_embedding.size(1) != self.embed_dim:
            projection = nn.Linear(object_embedding.size(1), self.embed_dim)
            object_embedding = projection(object_embedding)
        
        return object_embedding
    
    def store_prototype(self, 
                       class_name: str, 
                       image: torch.Tensor, 
                       mask: torch.Tensor,
                       confidence: float = 1.0):
        """Store object prototype in memory bank"""
        with torch.no_grad():
            object_embedding = self.extract_object_embeddings(image, mask)
            # Store in memory bank with confidence weighting
            self.memory_bank.store_or_update(
                class_name, object_embedding.squeeze(0), confidence
            )
    
    def retrieve_best_prototypes(self, 
                               query_image: torch.Tensor,
                               initial_mask: torch.Tensor,
                               class_name: str = None,
                               top_k: int = 3) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Retrieve best matching prototypes for query
        
        This implements the sophisticated retrieval mechanism from Section 3.4
        """
        with torch.no_grad():
            # Extract query object embedding
            query_embedding = self.extract_object_embeddings(query_image, initial_mask)
            
            if class_name:
                # Retrieve specific class prototype
                stored_embedding = self.memory_bank.retrieve(class_name)
                if stored_embedding is not None:
                    similarity = F.cosine_similarity(
                        query_embedding.squeeze(0), stored_embedding, dim=0
                    )
                    return [(class_name, stored_embedding, similarity.item())]
                return []
            else:
                # Retrieve similar prototypes across all classes
                return self.memory_bank.retrieve_similar(
                    query_embedding.squeeze(0), top_k=top_k
                )
    
    def adaptive_prototype_selection(self, 
                                   query_image: torch.Tensor,
                                   candidate_prototypes: List[Tuple[str, torch.Tensor, float]],
                                   confidence_threshold: float = 0.7) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Select prototypes adaptively based on confidence scores
        """
        if not candidate_prototypes:
            return []
        
        query_embedding = self.extract_object_embeddings(query_image, torch.ones_like(query_image[:,:1]))
        
        filtered_prototypes = []
        
        for class_name, proto_embedding, similarity in candidate_prototypes:
            # Compute confidence score
            combined_features = torch.cat([
                query_embedding.squeeze(0), 
                proto_embedding
            ], dim=0)
            
            confidence = self.confidence_scorer(combined_features).item()
            
            if confidence >= confidence_threshold:
                filtered_prototypes.append((class_name, proto_embedding, confidence))
        
        # Sort by confidence
        filtered_prototypes.sort(key=lambda x: x[2], reverse=True)
        
        return filtered_prototypes