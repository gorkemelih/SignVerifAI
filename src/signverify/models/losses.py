"""Loss functions for SignVerify.

Provides ContrastiveLoss and TripletLoss for Siamese network training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
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


class TripletLoss(nn.Module):
    """
    Triplet Loss for signature verification.
    
    L = max(0, d(a, p) - d(a, n) + margin)
    
    Where:
    - a = anchor embedding
    - p = positive embedding (same person)
    - n = negative embedding (different person)
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negative: Negative embeddings (B, D)
        
        Returns:
            Scalar loss value
        """
        pos_dist = torch.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.pairwise_distance(anchor, negative, p=2)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Mining Loss.
    
    For each anchor, select:
    - Hardest positive: max distance to positives
    - Hardest negative: min distance to negatives
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batch hard triplet loss.
        
        Args:
            embeddings: All embeddings in batch (B, D)
            labels: Person labels for each embedding (B,)
        
        Returns:
            Scalar loss value
        """
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        batch_size = embeddings.size(0)
        
        # Create masks for positives and negatives
        labels = labels.view(-1, 1)
        same_identity_mask = (labels == labels.T).float()
        
        # For each anchor, find hardest positive
        # Mask out different identities by setting to 0
        pos_distances = dist_mat * same_identity_mask
        hardest_positive_dist = pos_distances.max(dim=1)[0]
        
        # For each anchor, find hardest negative
        # Mask out same identities by setting to large value
        neg_distances = dist_mat + same_identity_mask * 1e6
        hardest_negative_dist = neg_distances.min(dim=1)[0]
        
        # Compute triplet loss
        loss = torch.clamp(
            hardest_positive_dist - hardest_negative_dist + self.margin,
            min=0.0
        )
        
        return loss.mean()


class ContrastiveLossWithHardMining(nn.Module):
    """
    Contrastive Loss with in-batch hard negative mining.
    """
    
    def __init__(self, margin: float = 0.5, hard_ratio: float = 0.3):
        super().__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss with hard negative emphasis.
        """
        # Euclidean distance
        distance = torch.pairwise_distance(emb1, emb2, p=2)
        
        # Standard contrastive loss
        loss_similar = target * torch.pow(distance, 2)
        loss_dissimilar = (1 - target) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        
        # Identify hard negatives: negative pairs with small distance
        hard_negative_mask = (target == 0) & (distance < self.margin)
        
        # Weight hard negatives more
        weights = torch.ones_like(target)
        weights[hard_negative_mask] = 2.0  # Double weight for hard negatives
        
        loss = 0.5 * (loss_similar + loss_dissimilar) * weights
        
        return loss.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss: Combination of Contrastive and Triplet Loss.
    
    For pairs data, constructs triplets within batch and combines losses.
    loss = alpha * contrastive + (1-alpha) * triplet
    """
    
    def __init__(
        self, 
        contrastive_margin: float = 0.5,
        triplet_margin: float = 0.2,
        alpha: float = 0.5,
        use_hard_negatives: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.contrastive_margin = contrastive_margin
        self.triplet_margin = triplet_margin
        self.use_hard_negatives = use_hard_negatives
        
        if use_hard_negatives:
            self.contrastive = ContrastiveLossWithHardMining(margin=contrastive_margin)
        else:
            self.contrastive = ContrastiveLoss(margin=contrastive_margin)
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid loss from pairs.
        
        For triplet component, we construct triplets from batch:
        - Anchors: emb1 where target=1 (genuine pairs)
        - Positives: emb2 where target=1
        - Negatives: emb2 where target=0 (hardest ones)
        """
        # Contrastive loss (always computed)
        contrastive_loss = self.contrastive(emb1, emb2, target)
        
        # Triplet loss - construct triplets from batch
        triplet_loss = self._compute_triplet_from_pairs(emb1, emb2, target)
        
        # Combine losses
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * triplet_loss
        
        return total_loss
    
    def _compute_triplet_from_pairs(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Construct triplets from pair batch and compute triplet loss."""
        # Positive pairs: target == 1
        pos_mask = target == 1
        # Negative pairs: target == 0
        neg_mask = target == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=emb1.device)
        
        # Anchors and positives from positive pairs
        anchors = emb1[pos_mask]
        positives = emb2[pos_mask]
        
        # Get negative embeddings
        neg_emb2 = emb2[neg_mask]
        
        n_anchors = anchors.size(0)
        n_negatives = neg_emb2.size(0)
        
        if n_negatives == 0:
            return torch.tensor(0.0, device=emb1.device)
        
        # For each anchor, find hardest negative (highest similarity = smallest distance)
        # Compute distances from each anchor to all negatives
        anchor_expanded = anchors.unsqueeze(1)  # (n_anchors, 1, D)
        neg_expanded = neg_emb2.unsqueeze(0)  # (1, n_negatives, D)
        
        distances = torch.norm(anchor_expanded - neg_expanded, p=2, dim=2)  # (n_anchors, n_negatives)
        
        # Hardest negative: minimum distance
        hardest_neg_idx = distances.argmin(dim=1)
        negatives = neg_emb2[hardest_neg_idx]
        
        # Compute triplet loss
        pos_dist = torch.pairwise_distance(anchors, positives, p=2)
        neg_dist = torch.pairwise_distance(anchors, negatives, p=2)
        
        loss = torch.clamp(pos_dist - neg_dist + self.triplet_margin, min=0.0)
        
        return loss.mean()


def get_loss_function(
    loss_type: str = "hybrid",
    margin: float = 0.5,
    triplet_margin: float = 0.2,
    use_hard_negatives: bool = True,
    hybrid_alpha: float = 0.5,
) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: "contrastive", "triplet", or "hybrid"
        margin: Margin for contrastive loss
        triplet_margin: Margin for triplet loss
        use_hard_negatives: Use hard negative mining
        hybrid_alpha: Weight for contrastive in hybrid
    
    Returns:
        Loss module
    """
    if loss_type == "contrastive":
        if use_hard_negatives:
            return ContrastiveLossWithHardMining(margin=margin)
        return ContrastiveLoss(margin=margin)
    elif loss_type == "triplet":
        return TripletLoss(margin=triplet_margin)
    elif loss_type == "hybrid":
        return HybridLoss(
            contrastive_margin=margin,
            triplet_margin=triplet_margin,
            alpha=hybrid_alpha,
            use_hard_negatives=use_hard_negatives,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
