"""Training module for SignVerify.

Handles model training with freeze/unfreeze, AMP, OneCycleLR/CosineAnnealing, and reduced logging.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from signverify.config import Config, PathConfig
from signverify.data.datasets import get_dataloaders
from signverify.models.losses import get_loss_function
from signverify.models.metrics import MetricTracker, VerificationMetrics
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


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters."""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters."""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = True


def get_param_groups(model: nn.Module, base_lr: float, backbone_lr_multiplier: float) -> list:
    """Get parameter groups with different learning rates."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_multiplier},
        {'params': head_params, 'lr': base_lr},
    ]


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats
        if len(batch) == 4:  # With index from HardNegativeDataset
            img1, img2, target, _ = batch
        else:
            img1, img2, target = batch
        
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_amp and device.type == 'cuda':
            with autocast():
                emb1, emb2, similarity = model(img1, img2)
                loss = criterion(emb1, emb2, target)
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            emb1, emb2, similarity = model(img1, img2)
            loss = criterion(emb1, emb2, target)
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, VerificationMetrics]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                img1, img2, target, _ = batch
            else:
                img1, img2, target = batch
            
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


def save_train_history(history: list, path: Path) -> None:
    """Save training history to CSV."""
    if not history:
        return
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def run_training(
    config: Optional[Config] = None,
    use_hard_mining: bool = True,
    scheduler_type: str = "onecycle",
) -> TrainingState:
    """
    Run full training pipeline with advanced features.
    """
    if config is None:
        config = Config()
    
    # Setup
    set_seed(42)
    device = config.device.get_device()
    paths = config.paths
    train_config = config.train
    
    # Print configuration once
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {train_config.epochs}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Loss: {train_config.loss_type} (margin={train_config.loss_margin})")
    logger.info(f"Scheduler: {scheduler_type}")
    logger.info(f"Freeze epochs: {train_config.freeze_backbone_epochs}")
    logger.info(f"AMP: {train_config.use_amp and device.type == 'cuda'}")
    logger.info(f"Log every: {train_config.log_every} epochs")
    logger.info("=" * 60)
    
    # Create model
    model = SiameseNetwork(
        embedding_dim=train_config.embedding_dim,
        pretrained=train_config.pretrained,
    )
    model = model.to(device)
    
    # Compile if enabled
    if train_config.use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Freeze backbone initially
    if train_config.freeze_backbone_epochs > 0:
        freeze_backbone(model)
        logger.info(f"Backbone frozen for first {train_config.freeze_backbone_epochs} epochs")
    
    # Loss function
    criterion = get_loss_function(
        loss_type=train_config.loss_type,
        margin=train_config.loss_margin,
        triplet_margin=train_config.triplet_margin,
        use_hard_negatives=train_config.use_hard_negatives,
    )
    
    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        paths=paths,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        use_hard_mining=use_hard_mining,
    )
    
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    # Initial optimizer - only with trainable parameters
    # During freeze phase, only head params are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    
    # Scheduler setup (will be recreated after unfreeze if needed)
    total_steps = len(train_loader) * train_config.epochs
    
    if scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=train_config.max_lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=1000,
        )
        step_per_batch = True
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
            eta_min=train_config.min_lr,
        )
        step_per_batch = False
    
    # AMP scaler
    scaler = GradScaler() if train_config.use_amp and device.type == 'cuda' else None
    
    # Training state
    state = TrainingState()
    history = []
    
    # Create run directory
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = paths.models / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {
        "epochs": train_config.epochs,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "loss_type": train_config.loss_type,
        "loss_margin": train_config.loss_margin,
        "scheduler": scheduler_type,
        "freeze_epochs": train_config.freeze_backbone_epochs,
        "use_amp": train_config.use_amp,
        "device": str(device),
    }
    write_json(config_dict, run_dir / "config.json")
    
    # Training loop
    logger.info("\nStarting training...")
    
    for epoch in range(1, train_config.epochs + 1):
        state.epoch = epoch
        
        # Unfreeze backbone after freeze epochs and rebuild optimizer
        if epoch == train_config.freeze_backbone_epochs + 1:
            unfreeze_backbone(model)
            
            # Rebuild optimizer with param groups (backbone + head with different LRs)
            param_groups = get_param_groups(
                model, 
                train_config.learning_rate, 
                train_config.backbone_lr_multiplier
            )
            optimizer = AdamW(param_groups, weight_decay=train_config.weight_decay)
            
            # Rebuild scheduler for remaining epochs
            remaining_steps = len(train_loader) * (train_config.epochs - epoch + 1)
            if scheduler_type == "onecycle":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[train_config.max_lr * train_config.backbone_lr_multiplier, train_config.max_lr],
                    total_steps=remaining_steps,
                    pct_start=0.1,
                )
                step_per_batch = True
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=train_config.epochs - epoch + 1,
                    eta_min=train_config.min_lr,
                )
                step_per_batch = False
            
            logger.info(f"Epoch {epoch}: Backbone unfrozen, optimizer rebuilt")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, train_config.gradient_clip,
            train_config.use_amp and device.type == 'cuda',
            scaler
        )
        
        # Step scheduler
        if step_per_batch:
            pass  # OneCycleLR steps per batch in train_epoch
        else:
            scheduler.step()
        
        # Validate
        val_loss, metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Current LR
        current_lr = optimizer.param_groups[-1]['lr']
        
        # Record history
        history_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'auc': round(metrics.auc, 4),
            'eer': round(metrics.eer, 4),
            'f1': round(metrics.f1, 4),
            'precision': round(metrics.precision, 4),
            'recall': round(metrics.recall, 4),
            'accuracy': round(metrics.accuracy, 4),
            'lr': round(current_lr, 8),
        }
        history.append(history_record)
        
        # Track
        state.train_losses.append(train_loss)
        state.val_losses.append(val_loss)
        state.val_aucs.append(metrics.auc)
        state.val_f1s.append(metrics.f1)
        
        # Log (every N epochs or on improvement)
        should_log = (epoch % train_config.log_every == 0) or (epoch == 1) or (epoch == train_config.epochs)
        
        if should_log:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"AUC: {metrics.auc:.4f} | EER: {metrics.eer:.4f} | "
                f"F1: {metrics.f1:.4f} | Acc: {metrics.accuracy:.4f} | "
                f"LR: {current_lr:.2e}"
            )
        
        # Check for best model
        if metrics.auc > state.best_auc:
            state.best_auc = metrics.auc
            state.best_f1 = metrics.f1
            state.best_epoch = epoch
            state.early_stop_counter = 0
            
            save_checkpoint(model, optimizer, state, run_dir / "checkpoint_best.pt", config)
            
            if not should_log:
                logger.info(f"Epoch {epoch:3d} | New best AUC: {metrics.auc:.4f}")
        else:
            state.early_stop_counter += 1
        
        # Early stopping
        if state.early_stop_counter >= train_config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save final checkpoint and history
    save_checkpoint(model, optimizer, state, run_dir / "checkpoint_latest.pt", config)
    save_train_history(history, run_dir / "train_history.csv")
    
    # Also save to reports
    save_train_history(history, paths.reports / "train_history.csv")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best AUC: {state.best_auc:.4f} at epoch {state.best_epoch}")
    logger.info(f"Best F1:  {state.best_f1:.4f}")
    logger.info(f"Checkpoints: {run_dir}")
    logger.info("=" * 60)
    
    return state
