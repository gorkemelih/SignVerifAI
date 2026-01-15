"""PyTorch Dataset classes for SignVerify.

Provides SignaturePairDataset for Siamese network training with augmentation.
"""

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from signverify.config import PathConfig
from signverify.utils.io import read_csv
from signverify.utils.logging import get_logger

logger = get_logger(__name__)


def grayscale_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale tensor to RGB by repeating channels.
    
    Args:
        img: Grayscale tensor of shape (1, H, W) or (H, W)
    
    Returns:
        RGB tensor of shape (3, H, W)
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img


class TrainTransform:
    """
    Training transform with augmentation.
    
    Applies:
    - Random rotation ±3°
    - Random translation 2-4%
    - Random brightness/contrast adjustment
    - Optional small noise
    - NO horizontal flip (signatures are directional)
    """
    
    def __init__(self):
        # Geometric augmentation
        self.affine = transforms.RandomAffine(
            degrees=3,  # ±3° rotation
            translate=(0.04, 0.04),  # 2-4% translation
            scale=None,
            shear=None,
            fill=255,  # White background
        )
        
        # Color augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
        )
        
        # ImageNet normalization
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply training augmentation."""
        # Convert to RGB for color jitter
        if img.mode == "L":
            img = img.convert("RGB")
        
        # Apply augmentations
        img = self.affine(img)
        img = self.color_jitter(img)
        
        # Convert to tensor
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Handle grayscale vs RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        else:
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        
        tensor = torch.from_numpy(arr)
        
        # Normalize
        mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.normalize_std).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor


class ValTransform:
    """
    Validation/Test transform (no augmentation).
    """
    
    def __init__(self):
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply validation transform (no augmentation)."""
        # Convert to numpy
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # To tensor (1, H, W) for grayscale
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        
        # Convert to RGB (3, H, W)
        tensor = grayscale_to_rgb(tensor)
        
        # Normalize with ImageNet stats
        mean = torch.tensor(self.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.normalize_std).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor


def default_transform(img: Image.Image) -> torch.Tensor:
    """Default transform for backward compatibility."""
    return ValTransform()(img)


class SignaturePairDataset(Dataset):
    """
    Dataset for Siamese signature verification.
    
    Loads pairs of images with similarity labels.
    Supports augmentation for training.
    """
    
    def __init__(
        self,
        pairs_csv: Path,
        data_dir: Path,
        transform: Optional[Callable] = None,
        is_train: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            pairs_csv: Path to pairs CSV file
            data_dir: Base directory for images (data_processed)
            transform: Optional transform function (overrides default)
            is_train: If True, use training augmentation
        """
        self.pairs_df = read_csv(pairs_csv)
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = TrainTransform()
        else:
            self.transform = ValTransform()
        
        logger.info(
            f"Loaded {len(self.pairs_df)} pairs from {pairs_csv} "
            f"(train={is_train}, augmentation={'on' if is_train else 'off'})"
        )
    
    def __len__(self) -> int:
        return len(self.pairs_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images with label.
        
        Returns:
            Tuple of (img1, img2, target) where target is 1 for similar, 0 for dissimilar
        """
        row = self.pairs_df.iloc[idx]
        
        img1_path = self.data_dir / row["img1_path"]
        img2_path = self.data_dir / row["img2_path"]
        
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        target = torch.tensor(row["target"], dtype=torch.float32)
        
        return img1, img2, target


class SignatureDataset(Dataset):
    """
    Simple dataset for single image classification/embedding.
    
    Useful for evaluation and inference.
    """
    
    def __init__(
        self,
        metadata_csv: Path,
        data_dir: Path,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            metadata_csv: Path to metadata or split CSV
            data_dir: Base directory for images
            transform: Optional transform function
        """
        self.df = read_csv(metadata_csv)
        self.data_dir = Path(data_dir)
        self.transform = transform or ValTransform()
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        """
        Get an image with metadata.
        
        Returns:
            Tuple of (image_tensor, person_id, label)
        """
        row = self.df.iloc[idx]
        
        img_path = self.data_dir / row["path"]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        
        return img, row["person_id"], row["label"]


def get_dataloaders(
    paths: Optional[PathConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get train and validation DataLoaders.
    
    Args:
        paths: Path configuration
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if paths is None:
        paths = PathConfig()
    
    train_dataset = SignaturePairDataset(
        pairs_csv=paths.pairs / "pairs_train.csv",
        data_dir=paths.data_processed,
        is_train=True,  # Enable augmentation for training
    )
    
    val_dataset = SignaturePairDataset(
        pairs_csv=paths.pairs / "pairs_val.csv",
        data_dir=paths.data_processed,
        is_train=False,  # No augmentation for validation
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
