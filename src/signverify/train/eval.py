"""Evaluation module for SignVerify.

Handles model evaluation on test set with comprehensive metrics, threshold tuning, and visualization.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
)
from torch.utils.data import DataLoader

from signverify.config import Config, PathConfig
from signverify.data.datasets import SignaturePairDataset
from signverify.models.metrics import MetricTracker, VerificationMetrics, log_metrics_summary
from signverify.models.siamese import SiameseNetwork
from signverify.utils.io import read_csv, write_json, write_markdown
from signverify.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    embedding_dim: int = 128,
) -> SiameseNetwork:
    """Load model from checkpoint with correct backbone architecture."""
    import json
    
    # Try to read backbone from config.json in same directory
    config_path = checkpoint_path.parent / "config.json"
    backbone_name = "mobilenet_v3_large"  # default
    
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
            backbone_name = config_data.get("backbone", "mobilenet_v3_large")
            logger.info(f"Loaded config: backbone={backbone_name}")
    
    model = SiameseNetwork(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path} (backbone={backbone_name})")
    return model


def evaluate_test_set(
    model: SiameseNetwork,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on test set."""
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


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict:
    """
    Find optimal thresholds for different objectives.
    
    Returns dict with:
    - eer: EER threshold
    - accuracy: Accuracy-maximizing threshold
    - f1: F1-maximizing threshold
    """
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # EER threshold (where FPR = 1 - TPR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # Search for best accuracy and F1 thresholds
    best_acc = 0
    best_acc_threshold = 0.5
    best_f1 = 0
    best_f1_threshold = 0.5
    
    # Sample thresholds to search
    search_thresholds = np.linspace(0.1, 0.99, 100)
    
    for thresh in search_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if acc > best_acc:
            best_acc = acc
            best_acc_threshold = thresh
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = thresh
    
    # Security threshold: Find threshold where FRR = 1% (very low false reject)
    # This means we accept more, but we want to see FAR at this point
    target_frr = 0.01
    security_threshold = 0.5
    security_far = 1.0
    
    for thresh in search_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        # Genuine = 1, Impostor = 0
        # FRR = FN / (FN + TP) = FN / total_genuine
        # FAR = FP / (FP + TN) = FP / total_impostor
        cm = sk_confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            total_genuine = fn + tp
            total_impostor = fp + tn
            
            if total_genuine > 0 and total_impostor > 0:
                frr = fn / total_genuine
                far = fp / total_impostor
                
                # Find threshold closest to target FRR
                if abs(frr - target_frr) < abs((security_threshold - 0.5) * 10):
                    if frr <= target_frr * 1.5:  # Allow some tolerance
                        security_threshold = thresh
                        security_far = far
    
    return {
        'eer': {'threshold': float(eer_threshold), 'eer': float(eer_value)},
        'accuracy': {'threshold': float(best_acc_threshold), 'accuracy': float(best_acc)},
        'f1': {'threshold': float(best_f1_threshold), 'f1': float(best_f1)},
        'security': {'threshold': float(security_threshold), 'far_at_frr_1pct': float(security_far)},
    }


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Compute classification metrics at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    cm = sk_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return ThresholdMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )


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


def plot_score_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Generate and save score distribution plot."""
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.6, label=f"Genuine (n={len(genuine_scores)})", color="green")
    plt.hist(impostor_scores, bins=50, alpha=0.6, label=f"Impostor (n={len(impostor_scores)})", color="red")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    metrics: ThresholdMetrics,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save confusion matrix plot."""
    cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ['Impostor', 'Genuine']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
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


def generate_eval_report(
    metrics: VerificationMetrics,
    thresholds: dict,
    threshold_metrics: dict,
    output_dir: Path,
    roc_path: Path,
    dist_path: Path,
) -> Path:
    """Generate comprehensive evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get metrics at each threshold
    eer_m = threshold_metrics['eer']
    acc_m = threshold_metrics['accuracy']
    f1_m = threshold_metrics['f1']
    
    md_content = f"""# Signature Verification - Evaluation Report

**Generated:** {timestamp}

## Core Verification Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | {metrics.auc:.4f} |
| **EER** | {metrics.eer:.4f} |
| **FAR @ FRR=1%** | {metrics.far_at_frr_01:.4f} |
| **FAR @ FRR=10%** | {metrics.far_at_frr_1:.4f} |

---

## Threshold Comparison

| Threshold Type | Threshold | Accuracy | Precision | Recall | F1 |
|----------------|-----------|----------|-----------|--------|-----|
| **EER** | {eer_m.threshold:.4f} | {eer_m.accuracy:.4f} | {eer_m.precision:.4f} | {eer_m.recall:.4f} | {eer_m.f1:.4f} |
| **Max Accuracy** | {acc_m.threshold:.4f} | {acc_m.accuracy:.4f} | {acc_m.precision:.4f} | {acc_m.recall:.4f} | {acc_m.f1:.4f} |
| **Max F1** | {f1_m.threshold:.4f} | {f1_m.accuracy:.4f} | {f1_m.precision:.4f} | {f1_m.recall:.4f} | {f1_m.f1:.4f} |

---

## Confusion Matrices

### At EER Threshold ({eer_m.threshold:.4f})

|  | Pred Impostor | Pred Genuine |
|---|---|---|
| **Actual Impostor** | TN={eer_m.tn} | FP={eer_m.fp} |
| **Actual Genuine** | FN={eer_m.fn} | TP={eer_m.tp} |

### At Max Accuracy Threshold ({acc_m.threshold:.4f})

|  | Pred Impostor | Pred Genuine |
|---|---|---|
| **Actual Impostor** | TN={acc_m.tn} | FP={acc_m.fp} |
| **Actual Genuine** | FN={acc_m.fn} | TP={acc_m.tp} |

### At Max F1 Threshold ({f1_m.threshold:.4f})

|  | Pred Impostor | Pred Genuine |
|---|---|---|
| **Actual Impostor** | TN={f1_m.tn} | FP={f1_m.fp} |
| **Actual Genuine** | FN={f1_m.fn} | TP={f1_m.tp} |

---

## Visualizations

### ROC Curve
![ROC Curve]({roc_path.name})

### Score Distribution
![Score Distribution]({dist_path.name})

---

## Recommendations

- **For security-critical applications:** Use EER threshold ({eer_m.threshold:.4f})
- **For best overall accuracy:** Use Max Accuracy threshold ({acc_m.threshold:.4f})
- **For balanced precision/recall:** Use Max F1 threshold ({f1_m.threshold:.4f})

### Model Quality Assessment

- **AUC = {metrics.auc:.4f}**: {"Excellent (>0.95)" if metrics.auc > 0.95 else "Very Good (>0.92)" if metrics.auc > 0.92 else "Good (>0.90)" if metrics.auc > 0.90 else "Moderate"}
- **Best Accuracy = {acc_m.accuracy:.4f}**: {"Excellent (>0.92)" if acc_m.accuracy > 0.92 else "Very Good (>0.90)" if acc_m.accuracy > 0.90 else "Good (>0.87)" if acc_m.accuracy > 0.87 else "Needs Improvement"}
- **Best F1 = {f1_m.f1:.4f}**: {"Excellent (>0.85)" if f1_m.f1 > 0.85 else "Very Good (>0.82)" if f1_m.f1 > 0.82 else "Good (>0.80)" if f1_m.f1 > 0.80 else "Moderate"}
"""
    
    md_path = output_dir / "eval_report.md"
    write_markdown(md_content, md_path)
    
    # JSON report with all data
    json_data = {
        "timestamp": timestamp,
        "core_metrics": {
            "auc": metrics.auc,
            "eer": metrics.eer,
            "far_at_frr_01": metrics.far_at_frr_01,
            "far_at_frr_1": metrics.far_at_frr_1,
        },
        "thresholds": thresholds,
        "threshold_metrics": {
            "eer": asdict(eer_m),
            "accuracy": asdict(acc_m),
            "f1": asdict(f1_m),
        },
    }
    json_path = output_dir / "eval_report.json"
    write_json(json_data, json_path)
    
    logger.info(f"Evaluation report saved to {output_dir}")
    
    return md_path


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    config: Optional[Config] = None,
    threshold_mode: Literal["eer", "accuracy", "f1", "all"] = "all",
) -> VerificationMetrics:
    """
    Run full evaluation pipeline with threshold tuning.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration
        threshold_mode: Which threshold(s) to report
    
    Returns:
        Evaluation metrics
    """
    if config is None:
        config = Config()
    
    paths = config.paths
    device = config.device.get_device()
    
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    # Find checkpoint
    if checkpoint_path is None:
        model_dirs = sorted(paths.models.glob("run_*"))
        if not model_dirs:
            raise FileNotFoundError("No training runs found.")
        
        latest_run = model_dirs[-1]
        checkpoint_path = latest_run / "checkpoint_best.pt"
        
        if not checkpoint_path.exists():
            checkpoint_path = latest_run / "checkpoint_latest.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, device, config.train.embedding_dim)
    
    # Create test pairs
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
    
    # Compute core metrics
    tracker = MetricTracker()
    tracker.y_true = y_true.tolist()
    tracker.y_scores = y_scores.tolist()
    metrics = tracker.compute()
    
    # Find optimal thresholds
    logger.info("Finding optimal thresholds...")
    thresholds = find_optimal_thresholds(y_true, y_scores)
    
    # Compute metrics at each threshold
    threshold_metrics = {
        'eer': compute_metrics_at_threshold(y_true, y_scores, thresholds['eer']['threshold']),
        'accuracy': compute_metrics_at_threshold(y_true, y_scores, thresholds['accuracy']['threshold']),
        'f1': compute_metrics_at_threshold(y_true, y_scores, thresholds['f1']['threshold']),
    }
    
    # Log results
    log_metrics_summary(metrics)
    
    logger.info("\nThreshold Comparison:")
    logger.info(f"  EER Threshold:      {thresholds['eer']['threshold']:.4f} -> Acc: {threshold_metrics['eer'].accuracy:.4f}")
    logger.info(f"  Max Acc Threshold:  {thresholds['accuracy']['threshold']:.4f} -> Acc: {threshold_metrics['accuracy'].accuracy:.4f}")
    logger.info(f"  Max F1 Threshold:   {thresholds['f1']['threshold']:.4f} -> F1: {threshold_metrics['f1'].f1:.4f}")
    
    # Generate visualizations
    paths.reports.mkdir(parents=True, exist_ok=True)
    
    roc_path = paths.reports / "roc_curve.png"
    dist_path = paths.reports / "score_distribution.png"
    
    plot_roc_curve(y_true, y_scores, roc_path, metrics.auc)
    plot_score_distribution(y_true, y_scores, dist_path)
    
    # Confusion matrix at best accuracy threshold
    cm_path = paths.reports / "confusion_matrix.png"
    plot_confusion_matrix(threshold_metrics['accuracy'], cm_path, f"Confusion Matrix @ Max Accuracy ({threshold_metrics['accuracy'].threshold:.4f})")
    
    # Generate report
    generate_eval_report(metrics, thresholds, threshold_metrics, paths.reports, roc_path, dist_path)
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return metrics
