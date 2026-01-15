"""Backbone network module for SignVerify.

MobileNetV3-Large with embedding layer for signature verification.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

from signverify.utils.logging import get_logger

logger = get_logger(__name__)


class SignatureBackbone(nn.Module):
    """
    MobileNetV3-Large backbone with embedding output.
    
    Attributes:
        embedding_dim: Output embedding dimension
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        """
        Initialize backbone.
        
        Args:
            embedding_dim: Dimension of output embedding
            pretrained: Use ImageNet pretrained weights
            freeze_bn: Freeze batch normalization layers
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pretrained MobileNetV3-Large
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.mobilenet_v3_large(weights=weights)
        
        # Remove classifier
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Get feature dimension (960 for MobileNetV3-Large)
        feature_dim = 960
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        
        # Optionally freeze batch norm
        if freeze_bn:
            self._freeze_bn()
        
        logger.info(
            f"Initialized SignatureBackbone: "
            f"embedding_dim={embedding_dim}, pretrained={pretrained}"
        )
    
    def _freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Embedding tensor of shape (B, embedding_dim)
        """
        # Features
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Embedding
        x = self.embedding(x)
        
        # L2 normalize for cosine similarity
        x = nn.functional.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


def create_backbone(
    embedding_dim: int = 128,
    pretrained: bool = True,
) -> SignatureBackbone:
    """
    Factory function to create backbone.
    
    Args:
        embedding_dim: Embedding dimension
        pretrained: Use pretrained weights
    
    Returns:
        SignatureBackbone instance
    """
    return SignatureBackbone(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )
