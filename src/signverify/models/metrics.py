"""Extended metrics module for SignVerify.

Includes F1, Precision, Recall, Confusion Matrix, Sensitivity, and more.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from signverify.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationMetrics:
    """Container for all verification metrics."""
    
    # ROC metrics
    auc: float = 0.0
    eer: float = 0.0
    eer_threshold: float = 0.0
    
    # FAR/FRR at specific points
    far_at_frr_01: float = 0.0  # FAR when FRR = 1%
    far_at_frr_1: float = 0.0   # FAR when FRR = 10%
    
    # Classification metrics (at EER threshold)
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    sensitivity: float = 0.0  # Same as recall (TPR)
    specificity: float = 0.0  # TNR
    
    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    # Additional
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0


@dataclass
class MetricTracker:
    """Track and compute verification metrics."""
    
    y_true: list = field(default_factory=list)
    y_scores: list = field(default_factory=list)
    
    def reset(self) -> None:
        """Reset tracked values."""
        self.y_true = []
        self.y_scores = []
    
    def update(self, y_true: np.ndarray, y_scores: np.ndarray) -> None:
        """Add batch of predictions."""
        self.y_true.extend(y_true.tolist() if hasattr(y_true, 'tolist') else y_true)
        self.y_scores.extend(y_scores.tolist() if hasattr(y_scores, 'tolist') else y_scores)
    
    def compute(self) -> VerificationMetrics:
        """Compute all metrics from tracked predictions."""
        y_true = np.array(self.y_true)
        y_scores = np.array(self.y_scores)
        
        if len(y_true) == 0:
            return VerificationMetrics()
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # EER calculation
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # FAR at specific FRR levels
        far_at_frr_01 = self._far_at_frr(fpr, fnr, 0.01)
        far_at_frr_1 = self._far_at_frr(fpr, fnr, 0.10)
        
        # Binary predictions at EER threshold
        y_pred = (y_scores >= eer_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Sensitivity (TPR) and Specificity (TNR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2
        
        return VerificationMetrics(
            auc=roc_auc,
            eer=eer,
            eer_threshold=eer_threshold,
            far_at_frr_01=far_at_frr_01,
            far_at_frr_1=far_at_frr_1,
            f1=f1,
            precision=precision,
            recall=recall,
            sensitivity=sensitivity,
            specificity=specificity,
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
        )
    
    def _far_at_frr(self, fpr: np.ndarray, fnr: np.ndarray, target_frr: float) -> float:
        """Get FAR at a specific FRR level."""
        idx = np.searchsorted(fnr[::-1], target_frr)
        idx = len(fnr) - idx - 1
        idx = max(0, min(idx, len(fpr) - 1))
        return fpr[idx]


def log_metrics_summary(metrics: VerificationMetrics, epoch: Optional[int] = None) -> None:
    """Log metrics in a formatted way."""
    prefix = f"Epoch {epoch} | " if epoch is not None else ""
    
    logger.info("=" * 50)
    logger.info(f"{prefix}VERIFICATION METRICS")
    logger.info("=" * 50)
    logger.info(f"AUC-ROC:     {metrics.auc:.4f}")
    logger.info(f"EER:         {metrics.eer:.4f} (threshold: {metrics.eer_threshold:.4f})")
    logger.info(f"FAR@FRR=1%:  {metrics.far_at_frr_01:.4f}")
    logger.info(f"FAR@FRR=10%: {metrics.far_at_frr_1:.4f}")
    logger.info("-" * 50)
    logger.info(f"F1 Score:    {metrics.f1:.4f}")
    logger.info(f"Precision:   {metrics.precision:.4f}")
    logger.info(f"Recall:      {metrics.recall:.4f}")
    logger.info(f"Sensitivity: {metrics.sensitivity:.4f}")
    logger.info(f"Specificity: {metrics.specificity:.4f}")
    logger.info(f"Accuracy:    {metrics.accuracy:.4f}")
    logger.info("-" * 50)
    logger.info(f"Confusion Matrix: TP={metrics.tp}, TN={metrics.tn}, FP={metrics.fp}, FN={metrics.fn}")
    logger.info("=" * 50)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: VerificationMetrics,
) -> None:
    """Log verbose epoch metrics."""
    logger.info("-" * 60)
    logger.info(f"EPOCH {epoch} SUMMARY")
    logger.info("-" * 60)
    logger.info(f"  Train Loss:    {train_loss:.4f}")
    logger.info(f"  Val Loss:      {val_loss:.4f}")
    logger.info(f"  AUC:           {metrics.auc:.4f}")
    logger.info(f"  EER:           {metrics.eer:.4f}")
    logger.info(f"  F1:            {metrics.f1:.4f}")
    logger.info(f"  Precision:     {metrics.precision:.4f}")
    logger.info(f"  Recall:        {metrics.recall:.4f}")
    logger.info(f"  Sensitivity:   {metrics.sensitivity:.4f}")
    logger.info(f"  FAR@FRR=10%:   {metrics.far_at_frr_1:.4f}")
    logger.info(f"  Confusion:     TP={metrics.tp} TN={metrics.tn} FP={metrics.fp} FN={metrics.fn}")
    logger.info("-" * 60)
