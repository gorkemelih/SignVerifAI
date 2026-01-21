"""Siamese Network module for SignVerify.

Twin network architecture for signature verification.
"""

import torch
import torch.nn as nn

from signverify.models.backbone import SignatureBackbone, create_backbone
from signverify.utils.logging import get_logger

logger = get_logger(__name__)


class SiameseNetwork(nn.Module):
    """
    Siamese Network for signature verification.
    
    Uses shared backbone to embed both images, then computes similarity.
    """
    
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        """
        Initialize Siamese network.
        
        Args:
            backbone_name: Architecture name (mobilenet_v3_large, resnet50, efficientnet_b0)
            embedding_dim: Embedding dimension from backbone
            pretrained: Use pretrained backbone weights
        """
        super().__init__()
        
        self.backbone = create_backbone(
            backbone_name=backbone_name,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )
        
        logger.info(f"Initialized SiameseNetwork: {backbone_name}, embedding_dim={embedding_dim}")
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single image.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Embedding of shape (B, embedding_dim)
        """
        return self.backbone(x)
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for image pair.
        
        Args:
            img1: First image tensor of shape (B, 3, H, W)
            img2: Second image tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple of (embedding1, embedding2, similarity)
            - embedding1/2: (B, embedding_dim)
            - similarity: (B,) cosine similarity scores
        """
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        
        # Cosine similarity (embeddings are already L2 normalized)
        similarity = torch.sum(emb1 * emb2, dim=1)
        
        return emb1, emb2, similarity
    
    def get_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Get similarity scores only (for inference).
        
        Returns:
            Similarity scores of shape (B,)
        """
        _, _, similarity = self.forward(img1, img2)
        return similarity
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for single image (for inference/clustering).
        
        Returns:
            Embedding of shape (B, embedding_dim)
        """
        return self.forward_one(x)


def create_siamese_network(
    backbone_name: str = "mobilenet_v3_large",
    embedding_dim: int = 128,
    pretrained: bool = True,
    device: torch.device = torch.device("cpu"),
) -> SiameseNetwork:
    """
    Factory function to create Siamese network.
    
    Args:
        backbone_name: Architecture name
        embedding_dim: Embedding dimension
        pretrained: Use pretrained backbone
        device: Target device
    
    Returns:
        SiameseNetwork on specified device
    """
    model = SiameseNetwork(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )
    return model.to(device)
