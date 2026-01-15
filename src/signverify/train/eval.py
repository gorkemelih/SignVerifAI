"""Evaluation module for SignVerify.

Handles model evaluation on test set with comprehensive metrics and visualization.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from signverify.config import Config, PathConfig
from signverify.data.datasets import SignaturePairDataset
from signverify.models.metrics import MetricTracker, VerificationMetrics, log_metrics_summary
from signverify.models.siamese import SiameseNetwork
from signverify.utils.io import read_csv, write_json, write_markdown
from signverify.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    embedding_dim: int = 128,
) -> SiameseNetwork:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device
        embedding_dim: Embedding dimension
    
    Returns:
        Loaded model
    """
    model = SiameseNetwork(embedding_dim=embedding_dim, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def evaluate_test_set(
    model: SiameseNetwork,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Compute device
    
    Returns:
        Tuple of (y_true, y_scores)
    """
    model.eval()
    
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for img1, img2, target in test_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            _, _, similarity = model(img1, img2)
            
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(similarity.cpu().numpy())
    
    return np.array(all_targets), np.array(all_scores)


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: Path,
    auc_value: float,
) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title("Signature Verification ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"ROC curve saved to {output_path}")


def plot_score_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Generate and save score distribution plot."""
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(
        genuine_scores,
        bins=50,
        alpha=0.6,
        label=f"Genuine (n={len(genuine_scores)})",
        color="green",
    )
    plt.hist(
        impostor_scores,
        bins=50,
        alpha=0.6,
        label=f"Impostor (n={len(impostor_scores)})",
        color="red",
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Score distribution saved to {output_path}")


def plot_confusion_matrix(
    metrics: VerificationMetrics,
    output_path: Path,
) -> None:
    """Generate and save confusion matrix plot."""
    cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix @ EER Threshold')
    plt.colorbar()
    
    classes = ['Impostor', 'Genuine']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def generate_eval_report(
    metrics: VerificationMetrics,
    output_dir: Path,
    roc_path: Path,
    dist_path: Path,
    cm_path: Path,
) -> Path:
    """Generate evaluation report in Markdown and JSON."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Markdown report
    md_content = f"""# Signature Verification - Evaluation Report

**Generated:** {timestamp}

## Metrics Summary

### Core Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | {metrics.auc:.4f} |
| **EER** | {metrics.eer:.4f} |
| **EER Threshold** | {metrics.eer_threshold:.4f} |
| **FAR @ FRR=1%** | {metrics.far_at_frr_01:.4f} |
| **FAR @ FRR=10%** | {metrics.far_at_frr_1:.4f} |

### Classification Metrics (at EER threshold)

| Metric | Value |
|--------|-------|
| **F1 Score** | {metrics.f1:.4f} |
| **Precision** | {metrics.precision:.4f} |
| **Recall** | {metrics.recall:.4f} |
| **Sensitivity (TPR)** | {metrics.sensitivity:.4f} |
| **Specificity (TNR)** | {metrics.specificity:.4f} |
| **Accuracy** | {metrics.accuracy:.4f} |
| **Balanced Accuracy** | {metrics.balanced_accuracy:.4f} |

### Confusion Matrix

|  | Predicted Impostor | Predicted Genuine |
|---|---|---|
| **Actual Impostor** | TN = {metrics.tn} | FP = {metrics.fp} |
| **Actual Genuine** | FN = {metrics.fn} | TP = {metrics.tp} |

## Visualizations

### ROC Curve

![ROC Curve]({roc_path.name})

### Score Distribution

![Score Distribution]({dist_path.name})

### Confusion Matrix

![Confusion Matrix]({cm_path.name})

## Interpretation

- **AUC = {metrics.auc:.4f}**: {"Excellent" if metrics.auc > 0.95 else "Good" if metrics.auc > 0.90 else "Moderate" if metrics.auc > 0.85 else "Needs Improvement"}
- **EER = {metrics.eer:.4f}**: Lower is better (equal false accept/reject rate)
- **F1 = {metrics.f1:.4f}**: Harmonic mean of precision and recall
"""
    
    md_path = output_dir / "eval_report.md"
    write_markdown(md_content, md_path)
    
    # JSON report
    json_data = {
        "timestamp": timestamp,
        **asdict(metrics),
    }
    json_path = output_dir / "eval_report.json"
    write_json(json_data, json_path)
    
    logger.info(f"Evaluation report saved to {output_dir}")
    
    return md_path


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    config: Optional[Config] = None,
) -> VerificationMetrics:
    """
    Run full evaluation pipeline with comprehensive metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint (auto-detect if None)
        config: Configuration (uses defaults if None)
    
    Returns:
        Evaluation metrics
    """
    if config is None:
        config = Config()
    
    paths = config.paths
    device = config.device.get_device()
    
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 60)
    
    # Find checkpoint
    if checkpoint_path is None:
        # Look for best checkpoint in latest run
        model_dirs = sorted(paths.models.glob("run_*"))
        if not model_dirs:
            raise FileNotFoundError("No training runs found. Run 'signverify train' first.")
        
        latest_run = model_dirs[-1]
        checkpoint_path = latest_run / "checkpoint_best.pt"
        
        if not checkpoint_path.exists():
            checkpoint_path = latest_run / "checkpoint_latest.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, device, config.train.embedding_dim)
    
    # Create test pairs (if not exists)
    test_pairs_path = paths.pairs / "pairs_test.csv"
    
    if not test_pairs_path.exists():
        logger.info("Generating test pairs...")
        from signverify.data.pairs import generate_balanced_pairs
        
        test_df = read_csv(paths.splits / "test.csv")
        test_pairs = generate_balanced_pairs(test_df, target_pairs=1200, seed=42)
        test_pairs.to_csv(test_pairs_path, index=False)
    
    # Create test DataLoader
    test_dataset = SignaturePairDataset(
        pairs_csv=test_pairs_path,
        data_dir=paths.data_processed,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    
    # Evaluate
    y_true, y_scores = evaluate_test_set(model, test_loader, device)
    
    # Compute comprehensive metrics
    tracker = MetricTracker()
    tracker.y_true = y_true.tolist()
    tracker.y_scores = y_scores.tolist()
    metrics = tracker.compute()
    
    log_metrics_summary(metrics)
    
    # Generate visualizations
    paths.reports.mkdir(parents=True, exist_ok=True)
    
    roc_path = paths.reports / "roc_curve.png"
    dist_path = paths.reports / "score_distribution.png"
    cm_path = paths.reports / "confusion_matrix.png"
    
    plot_roc_curve(y_true, y_scores, roc_path, metrics.auc)
    plot_score_distribution(y_true, y_scores, dist_path)
    plot_confusion_matrix(metrics, cm_path)
    
    # Generate report
    generate_eval_report(metrics, paths.reports, roc_path, dist_path, cm_path)
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return metrics
