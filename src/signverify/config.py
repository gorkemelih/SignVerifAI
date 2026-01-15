"""Configuration module for SignVerify pipeline.

Contains dataclasses for paths, training hyperparameters, and device settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


def _get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@dataclass
class PathConfig:
    """Path configuration for all project directories."""
    
    root: Path = field(default_factory=_get_project_root)
    
    @property
    def data_raw(self) -> Path:
        return self.root / "data_raw"
    
    @property
    def data_processed(self) -> Path:
        return self.root / "data_processed"
    
    @property
    def splits(self) -> Path:
        return self.root / "splits"
    
    @property
    def pairs(self) -> Path:
        return self.root / "pairs"
    
    @property
    def outputs(self) -> Path:
        return self.root / "outputs"
    
    @property
    def models(self) -> Path:
        return self.outputs / "models"
    
    @property
    def logs(self) -> Path:
        return self.outputs / "logs"
    
    @property
    def reports(self) -> Path:
        return self.outputs / "reports"
    
    @property
    def metadata_csv(self) -> Path:
        return self.data_processed / "metadata.csv"
    
    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for path in [self.splits, self.pairs, self.models, self.logs, self.reports]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    
    # Model
    backbone: str = "mobilenet_v3_large"
    embedding_dim: int = 128
    pretrained: bool = True
    
    # Loss
    loss_margin: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Training
    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 4
    
    # Validation
    val_frequency: int = 1
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_frequency: int = 5
    
    # Misc
    use_compile: bool = False  # torch.compile (optional)
    gradient_clip: float = 1.0


@dataclass
class DeviceConfig:
    """Device configuration with MPS support."""
    
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.device)
    
    @staticmethod
    def get_device_info() -> dict:
        """Get device availability info."""
        return {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        }


@dataclass
class SplitConfig:
    """Dataset split configuration."""
    
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    
    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class PairConfig:
    """Siamese pair generation configuration."""
    
    # Per genuine anchor sampling
    positive_per_anchor: int = 1  # genuine-genuine same person
    negative_same_person: int = 2  # genuine-forged same person
    negative_diff_person: int = 2  # genuine-genuine different person
    
    # Limits per split
    max_train_pairs: int = 50_000
    max_val_pairs: int = 10_000
    seed: int = 42


@dataclass
class Config:
    """Main configuration container."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    pair: PairConfig = field(default_factory=PairConfig)
    
    def __post_init__(self) -> None:
        self.paths.ensure_dirs()


# Global default config
def get_config() -> Config:
    """Get default configuration."""
    return Config()
