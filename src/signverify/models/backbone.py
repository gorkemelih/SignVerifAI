"""Backbone network module for SignVerify.

Supports MobileNetV3-Large, ResNet50, EfficientNet-B0 with embedding layer.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
)

from signverify.utils.logging import get_logger

logger = get_logger(__name__)


class SignatureBackbone(nn.Module):
    """
    Backbone with embedding output for signature verification.
    
    Supports multiple architectures:
    - mobilenet_v3_large: Fast, efficient (default)
    - resnet50: More capacity
    - efficientnet_b0: Good balance
    """
    
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name
        
        # Create backbone based on name
        if backbone_name == "mobilenet_v3_large":
            self._init_mobilenet(pretrained)
        elif backbone_name == "resnet50":
            self._init_resnet50(pretrained)
        elif backbone_name == "efficientnet_b0":
            self._init_efficientnet(pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Optionally freeze batch norm
        if freeze_bn:
            self._freeze_bn()
        
        logger.info(
            f"Initialized {backbone_name} backbone: "
            f"embedding_dim={embedding_dim}, pretrained={pretrained}"
        )
    
    def _init_mobilenet(self, pretrained: bool) -> None:
        """Initialize MobileNetV3-Large backbone."""
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.mobilenet_v3_large(weights=weights)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        feature_dim = 960
        
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.embedding_dim),
        )
    
    def _init_resnet50(self, pretrained: bool) -> None:
        """Initialize ResNet50 backbone."""
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.resnet50(weights=weights)
        
        # Remove final FC layer
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        self.avgpool = base_model.avgpool
        feature_dim = 2048
        
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, self.embedding_dim),
        )
    
    def _init_efficientnet(self, pretrained: bool) -> None:
        """Initialize EfficientNet-B0 backbone."""
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        feature_dim = 1280
        
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.embedding_dim),
        )
    
    def _freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            L2-normalized embedding of shape (B, embedding_dim)
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
    backbone_name: str = "mobilenet_v3_large",
    embedding_dim: int = 128,
    pretrained: bool = True,
) -> SignatureBackbone:
    """
    Factory function to create backbone.
    
    Args:
        backbone_name: Architecture name
        embedding_dim: Embedding dimension
        pretrained: Use pretrained weights
    
    Returns:
        SignatureBackbone instance
    """
    return SignatureBackbone(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )
