import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from iris_model import IrisModel
import logging
from sklearn.metrics.pairwise import cosine_similarity

class InferenceEngine:
    """
    Comprehensive inference engine implementing all Iris inference strategies:
    - One-shot inference
    - Context ensemble
    - Object-level context retrieval
    - In-context tuning
    """
    
    def __init__(self, 
                 model: IrisModel,
                 device: str = 'cuda'):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Task embedding cache for efficient inference
        self.task_embedding_cache = {}
        
        # Memory bank for seen classes (EMA-based)
        self.memory_bank = {}
        self.ema_alpha = 0.9  # EMA factor
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @torch.no_grad()
    def one_shot_inference(self,
                          query_image: torch.Tensor,
                          reference_image: torch.Tensor,
                          reference_mask: torch.Tensor,
                          apply_sigmoid: bool = True,
                          threshold: float = 0.5,
                          cache_task_embedding: bool = True,
                          task_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        One-shot inference with single reference example
        
        Args:
            query_image: Query image to segment (B, C, D, H, W)
            reference_image: Reference image (B, C, D, H, W)  
            reference_mask: Reference mask (B, 1, D, H, W)
            apply_sigmoid: Whether to apply sigmoid activation
            threshold: Threshold for binary prediction
            cache_task_embedding: Whether to cache task embedding for reuse
            task_id: Unique identifier for caching
            
        Returns:
            Dictionary with prediction results
        """
        query_image = query_image.to(self.device)
        reference_image = reference_image.to(self.device)
        reference_mask = reference_mask.to(self.device).float()
        self.model.eval()
        
        # Check cache first
        if task_id and task_id in self.task_embedding_cache:
            task_embedding = self.task_embedding_cache[task_id]
            self.logger.info(f"Using cached task embedding for {task_id}")
        else:
            # Encode task from reference
            task_embedding = self.model.encode_task(reference_image, reference_mask)
            
            # Cache for future use
            if cache_task_embedding and task_id:
                self.task_embedding_cache[task_id] = task_embedding.cpu()
        
        # Forward pass with task guidance
        logits = self.model(query_image, task_embedding=task_embedding)
        
        # Handle deep supervision outputs
        if isinstance(logits, list):
            logits = logits[-1]  # Use final output
        
        # Convert to probabilities and predictions
        if apply_sigmoid:
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        else:
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1, keepdim=True).float()
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': logits,
            'task_embedding': task_embedding
        }
    
    @torch.no_grad()
    def context_ensemble_inference(self,
                                 query_image: torch.Tensor,
                                 reference_images: List[torch.Tensor],
                                 reference_masks: List[torch.Tensor],
                                 ensemble_method: str = 'mean',
                                 weights: Optional[List[float]] = None,
                                 apply_sigmoid: bool = True,
                                 threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Context ensemble inference using multiple reference examples
        
        Args:
            query_image: Query image to segment
            reference_images: List of reference images
            reference_masks: List of reference masks
            ensemble_method: 'mean', 'weighted_mean', or 'max'
            weights: Weights for weighted ensemble (if None, uses equal weights)
            
        Returns:
            Dictionary with ensemble prediction results
        """
        self.model.eval()
        
        if len(reference_images) != len(reference_masks):
            raise ValueError("Number of reference images and masks must match")
        
        # Encode all reference examples
        task_embeddings = []
        for ref_img, ref_mask in zip(reference_images, reference_masks):
            task_emb = self.model.encode_task(ref_img, ref_mask)
            task_embeddings.append(task_emb)
        
        # Ensemble task embeddings
        task_embeddings = torch.stack(task_embeddings)  # (N, B, tokens, embed_dim)
        
        if ensemble_method == 'mean':
            ensemble_embedding = task_embeddings.mean(dim=0)
        elif ensemble_method == 'weighted_mean':
            if weights is None:
                weights = [1.0 / len(reference_images)] * len(reference_images)
            weights = torch.tensor(weights).to(self.device).view(-1, 1, 1, 1)
            ensemble_embedding = (task_embeddings * weights).sum(dim=0)
        elif ensemble_method == 'max':
            # Element-wise max pooling
            ensemble_embedding, _ = task_embeddings.max(dim=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
        
        # Segment query image with ensemble embedding
        logits = self.model(query_image, task_embedding=ensemble_embedding)
        
        if isinstance(logits, list):
            logits = logits[-1]
        
        # Convert to probabilities and predictions
        if apply_sigmoid:
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
        else:
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1, keepdim=True).float()
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': logits,
            'ensemble_embedding': ensemble_embedding,
            'individual_embeddings': task_embeddings
        }
    
    @torch.no_grad()
    def object_level_context_retrieval(self,
                                     query_image: torch.Tensor,
                                     reference_pool: List[Dict[str, torch.Tensor]],
                                     k_retrievals: int = 5,
                                     similarity_metric: str = 'cosine',
                                     apply_sigmoid: bool = True,
                                     threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Object-level context retrieval for fine-grained reference selection
        
        Args:
            query_image: Query image to segment
            reference_pool: List of dicts with 'image', 'mask', 'class_id' keys
            k_retrievals: Number of top references to retrieve per class
            similarity_metric: 'cosine' or 'euclidean'
            
        Returns:
            Dictionary with retrieval-based prediction results
        """
        self.model.eval()
        
        # Step 1: Get initial segmentation with random reference
        initial_ref = reference_pool[0]
        initial_result = self.one_shot_inference(
            query_image, initial_ref['image'], initial_ref['mask'],
            apply_sigmoid=False, cache_task_embedding=False
        )
        
        initial_logits = initial_result['logits']
        initial_probs = torch.sigmoid(initial_logits)
        initial_masks = (initial_probs > 0.5).float()
        
        # Step 2: Encode query-specific embeddings for each detected class
        query_features = self.model.encode_image(query_image)
        deep_features = query_features[-1]
        
        # Encode task embedding for initial prediction
        query_task_embeddings = []
        unique_classes = torch.unique(initial_masks)
        
        for class_val in unique_classes:
            if class_val == 0:  # Skip background
                continue
            class_mask = (initial_masks == class_val).float()
            if class_mask.sum() > 0:  # Only if class is present
                query_emb = self.model.task_encoder(deep_features, class_mask)
                query_task_embeddings.append(query_emb)
        
        if not query_task_embeddings:
            # Fall back to one-shot if no classes detected
            return initial_result
        
        # Step 3: Retrieve best references for each class
        retrieved_references = []
        
        for query_emb in query_task_embeddings:
            best_similarity = -float('inf')
            best_ref = None
            
            # Compare with all references in pool
            for ref_data in reference_pool:
                ref_emb = self.model.encode_task(ref_data['image'], ref_data['mask'])
                
                # Compute similarity between embeddings
                if similarity_metric == 'cosine':
                    # Flatten embeddings for cosine similarity
                    query_flat = query_emb.flatten()
                    ref_flat = ref_emb.flatten()
                    similarity = F.cosine_similarity(query_flat, ref_flat, dim=0)
                elif similarity_metric == 'euclidean':
                    similarity = -torch.norm(query_emb - ref_emb)
                else:
                    raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_ref = ref_data
            
            if best_ref:
                retrieved_references.append(best_ref)
        
        # Step 4: Final prediction with retrieved references
        if retrieved_references:
            ref_images = [ref['image'] for ref in retrieved_references]
            ref_masks = [ref['mask'] for ref in retrieved_references]
            
            final_result = self.context_ensemble_inference(
                query_image, ref_images, ref_masks,
                apply_sigmoid=apply_sigmoid, threshold=threshold
            )
            
            final_result['retrieved_references'] = retrieved_references
            final_result['initial_prediction'] = initial_result
            
            return final_result
        else:
            return initial_result
    
    def in_context_tuning(self,
                         query_image: torch.Tensor,
                         query_mask: torch.Tensor,
                         reference_image: torch.Tensor,
                         reference_mask: torch.Tensor,
                         tuning_steps: int = 10,
                         learning_rate: float = 1e-3,
                         apply_sigmoid: bool = True,
                         threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        In-context tuning by optimizing task embeddings while keeping model frozen
        
        Args:
            query_image: Query image
            query_mask: Query ground truth mask (for tuning)
            reference_image: Reference image
            reference_mask: Reference mask  
            tuning_steps: Number of optimization steps
            learning_rate: Learning rate for embedding optimization
            
        Returns:
            Dictionary with tuned prediction results
        """
        # Get initial task embedding
        with torch.no_grad():
            initial_task_embedding = self.model.encode_task(reference_image, reference_mask)
        
        # Create optimizable task embedding
        tuned_embedding = initial_task_embedding.clone().detach().requires_grad_(True)
        
        # Optimizer for task embedding only
        optimizer = torch.optim.Adam([tuned_embedding], lr=learning_rate)
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        self.model.train()  # Enable training mode for tuning
        
        # Optimization loop
        for step in range(tuning_steps):
            optimizer.zero_grad()
            
            # Forward pass with current embedding
            logits = self.model(query_image, task_embedding=tuned_embedding)
            
            # Compute loss
            loss = self.model.compute_loss(logits, query_mask)
            
            # Backward pass (only embedding gets gradients)
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                self.logger.info(f"Tuning step {step}, Loss: {loss.item():.4f}")
        
        # Final prediction with tuned embedding
        self.model.eval()
        with torch.no_grad():
            final_logits = self.model(query_image, task_embedding=tuned_embedding)
            
            if isinstance(final_logits, list):
                final_logits = final_logits[-1]
            
            if apply_sigmoid:
                probabilities = torch.sigmoid(final_logits)
                predictions = (probabilities > threshold).float()
            else:
                probabilities = F.softmax(final_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1, keepdim=True).float()
        
        # Restore model parameters
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': final_logits,
            'tuned_embedding': tuned_embedding.detach(),
            'initial_embedding': initial_task_embedding
        }
    
    def update_memory_bank(self, 
                          class_id: str, 
                          task_embedding: torch.Tensor):
        """
        Update memory bank with EMA for seen classes
        
        Args:
            class_id: Identifier for the class
            task_embedding: New task embedding to incorporate
        """
        if class_id in self.memory_bank:
            # EMA update
            old_embedding = self.memory_bank[class_id]
            new_embedding = (self.ema_alpha * old_embedding + 
                           (1 - self.ema_alpha) * task_embedding)
            self.memory_bank[class_id] = new_embedding
        else:
            # First time seeing this class
            self.memory_bank[class_id] = task_embedding.clone()
    
    def get_memory_bank_embedding(self, class_id: str) -> Optional[torch.Tensor]:
        """Get task embedding from memory bank for seen class"""
        return self.memory_bank.get(class_id, None)
    
    def clear_cache(self):
        """Clear task embedding cache"""
        self.task_embedding_cache.clear()
        self.logger.info("Task embedding cache cleared")
    
    def clear_memory_bank(self):
        """Clear memory bank"""
        self.memory_bank.clear()
        self.logger.info("Memory bank cleared")

# Test function
def test_inference_engine():
    """Test all inference strategies"""
    print("Testing Inference Engine...")
    
    # Create model and inference engine
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        embed_dim=128,
        num_classes=1
    )
    
    engine = InferenceEngine(model, device='cpu')
    
    # Create test data
    batch_size = 1
    size = (32, 32, 32)
    
    query_image = torch.randn(batch_size, 1, *size)
    reference_image = torch.randn(batch_size, 1, *size)
    reference_mask = torch.randint(0, 2, (batch_size, 1, *size)).float()
    
    print(f"Query image shape: {query_image.shape}")
    print(f"Reference image shape: {reference_image.shape}")
    print(f"Reference mask shape: {reference_mask.shape}")
    
    # Test one-shot inference
    print("\n1. Testing one-shot inference...")
    result1 = engine.one_shot_inference(
        query_image, reference_image, reference_mask,
        task_id="test_class_1"
    )
    print(f"One-shot prediction shape: {result1['predictions'].shape}")
    
    # Test context ensemble
    print("\n2. Testing context ensemble...")
    ref_images = [reference_image, reference_image]  # Duplicate for test
    ref_masks = [reference_mask, reference_mask]
    
    result2 = engine.context_ensemble_inference(
        query_image, ref_images, ref_masks
    )
    print(f"Ensemble prediction shape: {result2['predictions'].shape}")
    
    # Test object-level retrieval
    print("\n3. Testing object-level retrieval...")
    reference_pool = [
        {'image': reference_image, 'mask': reference_mask, 'class_id': 1},
        {'image': reference_image, 'mask': reference_mask, 'class_id': 1}
    ]
    
    result3 = engine.object_level_context_retrieval(
        query_image, reference_pool, k_retrievals=2
    )
    print(f"Retrieval prediction shape: {result3['predictions'].shape}")
    
    # Test in-context tuning
    print("\n4. Testing in-context tuning...")
    query_mask = torch.randint(0, 2, (batch_size, *size))
    
    result4 = engine.in_context_tuning(
        query_image, query_mask, reference_image, reference_mask,
        tuning_steps=5
    )
    print(f"Tuned prediction shape: {result4['predictions'].shape}")
    
    # Test memory bank
    print("\n5. Testing memory bank...")
    engine.update_memory_bank("class_1", result1['task_embedding'])
    cached_emb = engine.get_memory_bank_embedding("class_1")
    print(f"Cached embedding shape: {cached_emb.shape if cached_emb is not None else None}")
    
    print("\nInference engine test completed!")

if __name__ == "__main__":
    test_inference_engine()