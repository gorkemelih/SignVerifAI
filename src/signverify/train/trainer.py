"""Training module for SignVerify.

Handles model training with OneCycleLR, verbose metrics, and early stopping.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from signverify.config import Config, PathConfig
from signverify.data.datasets import get_dataloaders
from signverify.models.losses import ContrastiveLoss
from signverify.models.metrics import MetricTracker, VerificationMetrics, log_epoch_metrics
from signverify.models.siamese import SiameseNetwork
from signverify.utils.io import write_json
from signverify.utils.logging import get_logger
from signverify.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Track training state for checkpointing."""
    
    epoch: int = 0
    best_auc: float = 0.0
    best_f1: float = 0.0
    best_epoch: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    val_aucs: list = field(default_factory=list)
    val_f1s: list = field(default_factory=list)
    early_stop_counter: int = 0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (img1, img2, target) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        emb1, emb2, similarity = model(img1, img2)
        loss = criterion(emb1, emb2, target)
        
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Step scheduler per batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Log progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
    
    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, VerificationMetrics]:
    """
    Validate for one epoch with comprehensive metrics.
    
    Returns:
        Tuple of (average loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()
    
    with torch.no_grad():
        for img1, img2, target in val_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)
            
            emb1, emb2, similarity = model(img1, img2)
            loss = criterion(emb1, emb2, target)
            
            total_loss += loss.item()
            tracker.update(target.cpu().numpy(), similarity.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = tracker.compute()
    
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    path: Path,
    config: Config,
) -> None:
    """Save training checkpoint."""
    torch.save({
        "epoch": state.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_auc": state.best_auc,
        "best_f1": state.best_f1,
        "best_epoch": state.best_epoch,
        "train_losses": state.train_losses,
        "val_losses": state.val_losses,
        "val_aucs": state.val_aucs,
        "val_f1s": state.val_f1s,
    }, path)


def run_training(config: Optional[Config] = None) -> TrainingState:
    """
    Run full training pipeline.
    
    Args:
        config: Configuration (uses defaults if None)
    
    Returns:
        Final training state
    """
    if config is None:
        config = Config()
    
    # Setup
    set_seed(42)
    device = config.device.get_device()
    paths = config.paths
    train_config = config.train
    
    logger.info(f"Using device: {device}")
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Epochs: {train_config.epochs}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Learning rate: {train_config.learning_rate}")
    logger.info(f"Early stopping patience: {train_config.early_stopping_patience}")
    logger.info("Setting up training...")
    
    # Create model
    model = SiameseNetwork(
        embedding_dim=train_config.embedding_dim,
        pretrained=train_config.pretrained,
    )
    model = model.to(device)
    
    # Use torch.compile for PyTorch 2.0+ if enabled
    if train_config.use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        logger.info("Using torch.compile for acceleration")
    
    # Loss function
    criterion = ContrastiveLoss(margin=train_config.loss_margin)
    
    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        paths=paths,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    
    # OneCycleLR scheduler
    total_steps = len(train_loader) * train_config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # Max LR
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # Initial LR = max_lr / div_factor
        final_div_factor=1000,  # Final LR = max_lr / final_div_factor
    )
    logger.info(f"OneCycleLR: max_lr=1e-3, min_lr=1e-6, total_steps={total_steps}")
    
    # Training state
    state = TrainingState()
    
    # Create run directory
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = paths.models / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "epochs": train_config.epochs,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "embedding_dim": train_config.embedding_dim,
        "backbone": train_config.backbone,
        "loss_margin": train_config.loss_margin,
        "device": str(device),
        "scheduler": "OneCycleLR",
        "max_lr": 1e-3,
    }
    write_json(config_dict, run_dir / "config.json")
    
    # Training loop
    for epoch in range(1, train_config.epochs + 1):
        state.epoch = epoch
        
        logger.info(f"\nEpoch {epoch}/{train_config.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, train_config.gradient_clip
        )
        
        # Validate with full metrics
        val_loss, metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Log verbose metrics
        log_epoch_metrics(epoch, train_loss, val_loss, metrics)
        
        # Track history
        state.train_losses.append(train_loss)
        state.val_losses.append(val_loss)
        state.val_aucs.append(metrics.auc)
        state.val_f1s.append(metrics.f1)
        
        # Check for best model (using AUC as primary metric)
        if metrics.auc > state.best_auc:
            state.best_auc = metrics.auc
            state.best_f1 = metrics.f1
            state.best_epoch = epoch
            state.early_stop_counter = 0
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, state,
                run_dir / "checkpoint_best.pt", config
            )
            logger.info(f"  ✓ Saved best model (AUC: {metrics.auc:.4f}, F1: {metrics.f1:.4f})")
        else:
            state.early_stop_counter += 1
            logger.info(f"  No improvement. Early stop counter: {state.early_stop_counter}/{train_config.early_stopping_patience}")
        
        # Save checkpoint at intervals
        if epoch % train_config.checkpoint_frequency == 0:
            save_checkpoint(
                model, optimizer, state,
                run_dir / f"checkpoint_epoch_{epoch:03d}.pt", config
            )
        
        # Early stopping
        if state.early_stop_counter >= train_config.early_stopping_patience:
            logger.info(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            break
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, state,
        run_dir / "checkpoint_latest.pt", config
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best AUC: {state.best_auc:.4f} at epoch {state.best_epoch}")
    logger.info(f"Best F1:  {state.best_f1:.4f}")
    logger.info(f"Checkpoints saved to: {run_dir}")
    logger.info("=" * 60)
    
    return state
