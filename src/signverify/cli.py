"""CLI module for SignVerify.

Provides command-line interface using Typer.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from signverify.config import Config, DeviceConfig, PairConfig, PathConfig, SplitConfig, TrainConfig
from signverify.utils.logging import setup_logging

app = typer.Typer(
    name="signverify",
    help="Offline Signature Verification using Siamese Networks",
    add_completion=False,
)
console = Console()


def get_config(
    device: str = "auto",
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-4,
) -> Config:
    """Build configuration from CLI arguments."""
    return Config(
        paths=PathConfig(),
        train=TrainConfig(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=lr,
        ),
        device=DeviceConfig(device=device),
        split=SplitConfig(),
        pair=PairConfig(),
    )


@app.command()
def audit(
    check_duplicates: bool = typer.Option(True, help="Check for duplicate images"),
) -> None:
    """
    Audit dataset quality and generate reports.
    
    Checks: person counts, label distribution, corrupted images, duplicates.
    Outputs: outputs/reports/audit_report.md and audit_report.json
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Dataset Audit[/bold blue]")
    
    from signverify.data.audit import run_audit
    
    stats = run_audit()
    
    if stats.is_healthy:
        console.print("\n[bold green]✓ Dataset is healthy![/bold green]")
    else:
        console.print("\n[bold red]✗ Issues found - check reports[/bold red]")


@app.command()
def preprocess(
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing"),
) -> None:
    """
    Preprocess images (grayscale, crop, resize, cleanup).
    
    By default, skips already processed images (idempotent).
    Use --force to reprocess all.
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Preprocessing[/bold blue]")
    
    from signverify.data.preprocess import run_preprocess
    
    run_preprocess(force=force)
    
    console.print("\n[bold green]✓ Preprocessing complete![/bold green]")


@app.command()
def split(
    train_ratio: float = typer.Option(0.70, help="Training set ratio"),
    val_ratio: float = typer.Option(0.15, help="Validation set ratio"),
    test_ratio: float = typer.Option(0.15, help="Test set ratio"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """
    Split dataset into train/val/test (person-disjoint).
    
    Outputs: splits/train.csv, splits/val.csv, splits/test.csv
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Dataset Split[/bold blue]")
    
    from signverify.data.split import run_split
    
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    
    run_split(config=config)
    
    console.print("\n[bold green]✓ Split complete![/bold green]")


@app.command()
def pairs(
    pos_per_anchor: int = typer.Option(1, help="Positive pairs per anchor"),
    neg_same: int = typer.Option(2, help="Negative pairs (same person) per anchor"),
    neg_diff: int = typer.Option(2, help="Negative pairs (diff person) per anchor"),
    max_train: int = typer.Option(50000, help="Max train pairs"),
    max_val: int = typer.Option(10000, help="Max val pairs"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """
    Generate Siamese pairs for training.
    
    Outputs: pairs/pairs_train.csv, pairs/pairs_val.csv
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Pair Generation[/bold blue]")
    
    from signverify.data.pairs import run_pairs
    
    config = PairConfig(
        positive_per_anchor=pos_per_anchor,
        negative_same_person=neg_same,
        negative_diff_person=neg_diff,
        max_train_pairs=max_train,
        max_val_pairs=max_val,
        seed=seed,
    )
    
    run_pairs(config=config)
    
    console.print("\n[bold green]✓ Pair generation complete![/bold green]")


@app.command()
def train(
    device: str = typer.Option("auto", help="Device: auto, mps, cuda, cpu"),
    epochs: int = typer.Option(30, help="Number of epochs"),
    batch_size: int = typer.Option(128, help="Batch size"),
    embedding_dim: int = typer.Option(128, help="Embedding dimension"),
    pretrained: bool = typer.Option(True, help="Use pretrained backbone"),
    freeze_epochs: int = typer.Option(3, help="Freeze backbone for N epochs"),
    backbone_lr: float = typer.Option(1e-5, help="Backbone learning rate"),
    head_lr: float = typer.Option(3e-4, help="Head learning rate"),
    hard_mining: bool = typer.Option(True, help="Use hard negative mining"),
    loss: str = typer.Option("hybrid", help="Loss: contrastive, triplet, or hybrid"),
    scheduler: str = typer.Option("onecycle", help="LR scheduler: cosine or onecycle"),
    log_every: int = typer.Option(5, help="Log metrics every N epochs"),
    use_compile: bool = typer.Option(False, help="Use torch.compile (PyTorch 2.0+)"),
) -> None:
    """
    Train Siamese network for signature verification.
    
    Outputs: outputs/models/run_*/checkpoints
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Training[/bold blue]")
    console.print(f"Epochs: {epochs} | Batch: {batch_size} | Loss: {loss}")
    console.print(f"Backbone LR: {backbone_lr} | Head LR: {head_lr} | Freeze: {freeze_epochs} epochs")
    
    from signverify.train.trainer import run_training
    
    config = Config(
        paths=PathConfig(),
        train=TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone_epochs=freeze_epochs,
            loss_type=loss,
            use_hard_negatives=hard_mining,
            log_every=log_every,
            use_compile=use_compile,
        ),
        device=DeviceConfig(device=device),
    )
    
    state = run_training(config=config, use_hard_mining=hard_mining, scheduler_type=scheduler)
    
    console.print(f"\n[bold green]✓ Training complete! Best AUC: {state.best_auc:.4f}[/bold green]")


@app.command()
def eval(
    checkpoint: Optional[Path] = typer.Option(None, help="Checkpoint path (auto-detect if not set)"),
    device: str = typer.Option("auto", help="Device: auto, mps, cuda, cpu"),
    threshold_mode: str = typer.Option("all", help="Threshold mode: eer, accuracy, f1, or all"),
) -> None:
    """
    Evaluate trained model on test set with threshold tuning.
    
    Outputs: outputs/reports/eval_report.md, ROC curve, score distribution
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Evaluation[/bold blue]")
    console.print(f"Threshold mode: {threshold_mode}")
    
    from signverify.train.eval import run_evaluation
    
    config = Config(
        device=DeviceConfig(device=device),
    )
    
    metrics = run_evaluation(checkpoint_path=checkpoint, config=config, threshold_mode=threshold_mode)
    
    console.print(f"\n[bold green]✓ Evaluation complete! AUC: {metrics.auc:.4f}, EER: {metrics.eer:.4f}[/bold green]")


@app.command()
def info() -> None:
    """Show project info and device status."""
    console.print("[bold blue]SignVerify - Project Info[/bold blue]\n")
    
    from signverify import __version__
    from signverify.config import DeviceConfig, PathConfig
    
    paths = PathConfig()
    device_info = DeviceConfig.get_device_info()
    default_device = DeviceConfig().get_device()
    
    console.print(f"Version: {__version__}")
    console.print(f"Project root: {paths.root}")
    console.print(f"\nDevice Status:")
    console.print(f"  MPS available: {device_info['mps_available']}")
    console.print(f"  CUDA available: {device_info['cuda_available']}")
    console.print(f"  Default device: {default_device}")
    
    # Check data status
    console.print(f"\nData Status:")
    console.print(f"  Metadata: {'✓' if paths.metadata_csv.exists() else '✗'}")
    console.print(f"  Splits: {'✓' if (paths.splits / 'train.csv').exists() else '✗'}")
    console.print(f"  Pairs: {'✓' if (paths.pairs / 'pairs_train.csv').exists() else '✗'}")


@app.command()
def pipeline(
    device: str = typer.Option("auto", help="Device for training"),
    epochs: int = typer.Option(50, help="Number of epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
) -> None:
    """
    Run full pipeline: audit → split → pairs → train → eval
    """
    setup_logging()
    console.print("[bold blue]SignVerify - Full Pipeline[/bold blue]\n")
    
    from signverify.data.audit import run_audit
    from signverify.data.pairs import run_pairs
    from signverify.data.split import run_split
    from signverify.train.eval import run_evaluation
    from signverify.train.trainer import run_training
    
    # 1. Audit
    console.print("[bold]Step 1/5: Audit[/bold]")
    run_audit()
    
    # 2. Split
    console.print("\n[bold]Step 2/5: Split[/bold]")
    run_split()
    
    # 3. Pairs
    console.print("\n[bold]Step 3/5: Pairs[/bold]")
    run_pairs()
    
    # 4. Train
    console.print("\n[bold]Step 4/5: Train[/bold]")
    config = Config(
        train=TrainConfig(epochs=epochs, batch_size=batch_size),
        device=DeviceConfig(device=device),
    )
    run_training(config=config)
    
    # 5. Eval
    console.print("\n[bold]Step 5/5: Evaluate[/bold]")
    run_evaluation(config=config)
    
    console.print("\n[bold green]✓ Full pipeline complete![/bold green]")


if __name__ == "__main__":
    app()
