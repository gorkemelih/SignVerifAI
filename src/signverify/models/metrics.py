"""Metrics module for SignVerify.

Provides ROC-AUC, FAR/FRR, and EER calculation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from signverify.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationMetrics:
    """Container for verification metrics."""
    
    auc: float
    eer: float
    eer_threshold: float
    far_at_frr_01: float  # FAR when FRR = 1%
    far_at_frr_1: float   # FAR when FRR = 10%
    threshold_at_frr_01: float
    threshold_at_frr_1: float


def compute_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    EER is where FAR = FRR (False Accept Rate = False Reject Rate).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
    
    Returns:
        Tuple of (EER value, threshold at EER)
    """
    fnr = 1 - tpr  # False Negative Rate = 1 - TPR
    
    # Find where FAR ≈ FRR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return float(eer), float(eer_threshold)


def compute_far_at_frr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_frr: float,
) -> tuple[float, float]:
    """
    Compute FAR at a specific FRR level.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
        target_frr: Target False Reject Rate
    
    Returns:
        Tuple of (FAR at target FRR, threshold)
    """
    fnr = 1 - tpr
    
    # Find closest FRR to target
    idx = np.nanargmin(np.abs(fnr - target_frr))
    
    return float(fpr[idx]), float(thresholds[idx])


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> VerificationMetrics:
    """
    Compute all verification metrics.
    
    Args:
        y_true: Ground truth labels (1=similar, 0=dissimilar)
        y_scores: Predicted similarity scores
    
    Returns:
        VerificationMetrics dataclass
    """
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # AUC
    auc = float(roc_auc_score(y_true, y_scores))
    
    # EER
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    
    # FAR at specific FRR levels
    far_01, thresh_01 = compute_far_at_frr(fpr, tpr, thresholds, 0.01)
    far_1, thresh_1 = compute_far_at_frr(fpr, tpr, thresholds, 0.10)
    
    metrics = VerificationMetrics(
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
        far_at_frr_01=far_01,
        far_at_frr_1=far_1,
        threshold_at_frr_01=thresh_01,
        threshold_at_frr_1=thresh_1,
    )
    
    logger.info(f"Metrics: AUC={auc:.4f}, EER={eer:.4f} @ threshold={eer_threshold:.4f}")
    
    return metrics


def log_metrics_summary(metrics: VerificationMetrics) -> None:
    """Log metrics summary."""
    logger.info("=" * 50)
    logger.info("VERIFICATION METRICS")
    logger.info("=" * 50)
    logger.info(f"AUC-ROC:     {metrics.auc:.4f}")
    logger.info(f"EER:         {metrics.eer:.4f} (threshold: {metrics.eer_threshold:.4f})")
    logger.info(f"FAR@FRR=1%:  {metrics.far_at_frr_01:.4f}")
    logger.info(f"FAR@FRR=10%: {metrics.far_at_frr_1:.4f}")
    logger.info("=" * 50)


class MetricTracker:
    """Track and accumulate metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated values."""
        self.y_true: list[float] = []
        self.y_scores: list[float] = []
        self.losses: list[float] = []
    
    def update(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        loss: Optional[float] = None,
    ) -> None:
        """Update with batch values."""
        self.y_true.extend(y_true.tolist())
        self.y_scores.extend(y_scores.tolist())
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> VerificationMetrics:
        """Compute final metrics."""
        return compute_metrics(
            np.array(self.y_true),
            np.array(self.y_scores),
        )
    
    def get_avg_loss(self) -> float:
        """Get average loss."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)
