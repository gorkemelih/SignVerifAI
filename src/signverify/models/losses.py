"""Loss functions for SignVerify.

Provides ContrastiveLoss for Siamese network training.
"""

import torch
import torch.nn as nn

from signverify.utils.logging import get_logger

logger = get_logger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese networks.
    
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    Where:
    - Y = 1 for similar pairs (positive)
    - Y = 0 for dissimilar pairs (negative)
    - D = distance (1 - cosine_similarity for normalized embeddings)
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
        logger.info(f"Initialized ContrastiveLoss with margin={margin}")
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            emb1: First embedding (B, D), L2 normalized
            emb2: Second embedding (B, D), L2 normalized
            target: Target labels (B,), 1=similar, 0=dissimilar
        
        Returns:
            Scalar loss value
        """
        # Euclidean distance (for L2 normalized vectors)
        distance = torch.pairwise_distance(emb1, emb2, p=2)
        
        # Contrastive loss
        # For similar pairs (target=1): minimize distance
        # For dissimilar pairs (target=0): push apart beyond margin
        loss_similar = target * torch.pow(distance, 2)
        loss_dissimilar = (1 - target) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        
        loss = 0.5 * (loss_similar + loss_dissimilar)
        
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """
    Binary cross-entropy loss with cosine similarity.
    
    Alternative to ContrastiveLoss using BCE on similarity scores.
    """
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        similarity: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss on similarity scores.
        
        Args:
            similarity: Cosine similarity scores (B,)
            target: Target labels (B,), 1=similar, 0=dissimilar
        
        Returns:
            Scalar loss value
        """
        # Scale similarity from [-1, 1] to logits
        # similarity is already in [-1, 1] range
        logits = similarity * 5.0  # Scale for better gradients
        
        return self.bce(logits, target)


def get_loss_function(loss_type: str = "contrastive", margin: float = 1.0) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: "contrastive" or "cosine_bce"
        margin: Margin for contrastive loss
    
    Returns:
        Loss module
    """
    if loss_type == "contrastive":
        return ContrastiveLoss(margin=margin)
    elif loss_type == "cosine_bce":
        return CosineSimilarityLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
