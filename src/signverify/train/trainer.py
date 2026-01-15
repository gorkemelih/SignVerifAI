"""Training module for SignVerify.

Handles model training with validation, checkpointing, and logging.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from signverify.config import Config, DeviceConfig, PathConfig, TrainConfig
from signverify.data.datasets import get_dataloaders
from signverify.models.losses import ContrastiveLoss
from signverify.models.metrics import MetricTracker, compute_metrics
from signverify.models.siamese import SiameseNetwork, create_siamese_network
from signverify.utils.io import write_json
from signverify.utils.logging import get_logger, setup_logging
from signverify.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Training state container."""
    
    epoch: int = 0
    best_auc: float = 0.0
    best_epoch: int = 0
    train_losses: list = None
    val_aucs: list = None
    
    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_aucs is None:
            self.val_aucs = []


class Trainer:
    """
    Trainer for Siamese signature verification network.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Full configuration (uses defaults if None)
        """
        self.config = config or Config()
        self.paths = self.config.paths
        self.train_config = self.config.train
        
        # Setup device
        self.device = self.config.device.get_device()
        logger.info(f"Using device: {self.device}")
        
        # Setup reproducibility
        set_seed(self.config.split.seed)
        
        # Initialize components
        self.model: Optional[SiameseNetwork] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.criterion: Optional[ContrastiveLoss] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # State
        self.state = TrainingState()
        
        # Paths
        self.run_dir = self._create_run_dir()
    
    def _create_run_dir(self) -> Path:
        """Create unique run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.paths.models / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def setup(self) -> None:
        """Setup model, optimizer, criterion, and dataloaders."""
        logger.info("Setting up training...")
        
        # Model
        self.model = create_siamese_network(
            embedding_dim=self.train_config.embedding_dim,
            pretrained=self.train_config.pretrained,
            device=self.device,
        )
        
        # Optional: torch.compile (PyTorch 2.0+)
        if self.train_config.use_compile and hasattr(torch, "compile"):
            logger.info("Applying torch.compile...")
            self.model = torch.compile(self.model)
        
        # Criterion
        self.criterion = ContrastiveLoss(margin=self.train_config.loss_margin)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.epochs,
            eta_min=1e-6,
        )
        
        # DataLoaders
        self.train_loader, self.val_loader = get_dataloaders(
            paths=self.paths,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
        # Save config
        self._save_config()
    
    def _save_config(self) -> None:
        """Save training configuration."""
        config_dict = {
            "backbone": self.train_config.backbone,
            "embedding_dim": self.train_config.embedding_dim,
            "learning_rate": self.train_config.learning_rate,
            "batch_size": self.train_config.batch_size,
            "epochs": self.train_config.epochs,
            "loss_margin": self.train_config.loss_margin,
            "device": str(self.device),
        }
        write_json(config_dict, self.run_dir / "config.json")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        tracker = MetricTracker()
        
        for batch_idx, (img1, img2, target) in enumerate(self.train_loader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            target = target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            emb1, emb2, similarity = self.model(img1, img2)
            
            # Loss
            loss = self.criterion(emb1, emb2, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.train_config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Track
            tracker.update(
                target.cpu().numpy(),
                similarity.detach().cpu().numpy(),
                loss.item(),
            )
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        return tracker.get_avg_loss()
    
    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (val_loss, val_auc)
        """
        self.model.eval()
        tracker = MetricTracker()
        
        for img1, img2, target in self.val_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            target = target.to(self.device)
            
            emb1, emb2, similarity = self.model(img1, img2)
            loss = self.criterion(emb1, emb2, target)
            
            tracker.update(
                target.cpu().numpy(),
                similarity.cpu().numpy(),
                loss.item(),
            )
        
        metrics = tracker.compute()
        return tracker.get_avg_loss(), metrics.auc
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.state.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_auc": self.state.best_auc,
            "train_losses": self.state.train_losses,
            "val_aucs": self.state.val_aucs,
        }
        
        # Save latest
        torch.save(checkpoint, self.run_dir / "checkpoint_latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.run_dir / "checkpoint_best.pt")
            logger.info(f"  Saved best model (AUC: {self.state.best_auc:.4f})")
        
        # Periodic save
        if self.state.epoch % self.train_config.checkpoint_frequency == 0:
            torch.save(
                checkpoint,
                self.run_dir / f"checkpoint_epoch_{self.state.epoch:03d}.pt",
            )
    
    def run(self) -> TrainingState:
        """
        Run full training loop.
        
        Returns:
            Final training state
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        
        self.setup()
        
        no_improve_count = 0
        
        for epoch in range(1, self.train_config.epochs + 1):
            self.state.epoch = epoch
            
            logger.info(f"\nEpoch {epoch}/{self.train_config.epochs}")
            logger.info("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            self.state.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.train_config.val_frequency == 0:
                val_loss, val_auc = self.validate()
                self.state.val_aucs.append(val_auc)
                
                logger.info(
                    f"  Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
                )
                
                # Check improvement
                is_best = val_auc > self.state.best_auc
                if is_best:
                    self.state.best_auc = val_auc
                    self.state.best_epoch = epoch
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                self.save_checkpoint(is_best=is_best)
            
            # LR scheduler
            self.scheduler.step()
            
            # Early stopping
            if no_improve_count >= self.train_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best AUC: {self.state.best_auc:.4f} at epoch {self.state.best_epoch}")
        logger.info(f"Checkpoints saved to: {self.run_dir}")
        logger.info("=" * 60)
        
        return self.state


def run_training(config: Optional[Config] = None) -> TrainingState:
    """
    Entry point for training.
    
    Args:
        config: Training configuration
    
    Returns:
        Final training state
    """
    trainer = Trainer(config=config)
    return trainer.run()
