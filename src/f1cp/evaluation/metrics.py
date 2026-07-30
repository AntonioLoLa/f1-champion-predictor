"""The fuller evaluation metric suite, built on top of what training already scores.

``training.metrics`` computes exactly what model *selection* needs (a grouped log loss,
Brier score, ECE, top-1 hit rate). This module adds the standard classification-report
metrics PLAN.md asks for on top of that, reusing the same conventions: rank/probability
metrics (ROC AUC, PR AUC, log loss, calibration curve) score the **calibrated,
pre-normalisation** probability, pointwise; classification metrics (accuracy, precision,
recall, F1, confusion matrix) need a discrete predicted label, which only exists once
probabilities are turned into "who did the model pick as champion" via
:func:`~f1cp.training.metrics.top_ranked_mask` -- the same per-group argmax
:func:`~f1cp.training.metrics.top1_hit_rate` already uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from f1cp.training.metrics import top_ranked_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

__all__ = [
    "calibration_curve_points",
    "compute_full_metrics",
]

_EPS = 1e-15


def _predicted_and_true(
    frame: pl.DataFrame,
    *,
    normalized_column: str,
    target_column: str,
    group_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the true label and the per-group top-ranked predicted label."""
    mask = top_ranked_mask(frame, probability_column=normalized_column, group_columns=group_columns)
    y_pred = mask.to_numpy().astype(int)
    y_true = frame[target_column].to_numpy().astype(int)
    return y_true, y_pred


def _classification_metrics(
    frame: pl.DataFrame,
    *,
    normalized_column: str,
    target_column: str,
    group_columns: Sequence[str],
) -> dict[str, float | None]:
    if frame.is_empty():
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positive": None,
            "false_positive": None,
            "false_negative": None,
            "true_negative": None,
        }

    y_true, y_pred = _predicted_and_true(
        frame,
        normalized_column=normalized_column,
        target_column=target_column,
        group_columns=group_columns,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
    }


def _rank_metrics(
    frame: pl.DataFrame, *, calibrated_column: str, target_column: str
) -> dict[str, float | None]:
    if frame.is_empty():
        return {"roc_auc": None, "pr_auc": None, "log_loss": None}

    y_true = frame[target_column].to_numpy().astype(int)
    if len(np.unique(y_true)) < 2:
        # ROC AUC / PR AUC / log loss are undefined against a single-class slice (e.g. an
        # early-phase subset of a season where nobody there is the eventual champion).
        return {"roc_auc": None, "pr_auc": None, "log_loss": None}

    y_score = np.clip(frame[calibrated_column].to_numpy(), _EPS, 1.0 - _EPS)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
    }


def calibration_curve_points(
    frame: pl.DataFrame,
    *,
    calibrated_column: str,
    target_column: str,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Reliability-diagram points: per-bin predicted confidence vs. observed accuracy.

    Args:
        frame: Rows carrying ``calibrated_column`` and ``target_column``.
        calibrated_column: Pointwise calibrated probability column.
        target_column: Binary champion label.
        n_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        One dict per non-empty bin: ``bin_center``, ``confidence`` (mean predicted
        probability), ``accuracy`` (observed positive rate) and ``count``.
    """
    if frame.is_empty():
        return []

    p = frame[calibrated_column].to_numpy()
    y = frame[target_column].to_numpy().astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    points: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        points.append(
            {
                "bin_center": float((edges[b] + edges[b + 1]) / 2),
                "confidence": float(p[mask].mean()),
                "accuracy": float(y[mask].mean()),
                "count": float(count),
            }
        )
    return points


def compute_full_metrics(
    frame: pl.DataFrame,
    *,
    calibrated_column: str,
    normalized_column: str,
    target_column: str,
    group_columns: Sequence[str],
) -> dict[str, float | None]:
    """Compute the full classification + ranking metric suite for one slice of rows.

    Complements :func:`f1cp.training.metrics.compute_metrics`; called the same way
    (once pooled, once per phase, and here also once per season) so evaluation reports
    at every granularity training's selection metrics already do.

    Args:
        frame: Scored rows, carrying keys, ``target_column``, ``calibrated_column`` and
            ``normalized_column``.
        calibrated_column: Pointwise calibrated probability column.
        normalized_column: Per-group-normalised probability column.
        target_column: Binary champion label.
        group_columns: Columns identifying a mutually-exclusive group.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall``, ``f1``, ``true_positive``,
        ``false_positive``, ``false_negative``, ``true_negative``, ``roc_auc``, ``pr_auc``
        and ``log_loss``. Any value is ``None`` when it could not be computed on ``frame``.
    """
    return {
        **_classification_metrics(
            frame,
            normalized_column=normalized_column,
            target_column=target_column,
            group_columns=group_columns,
        ),
        **_rank_metrics(frame, calibrated_column=calibrated_column, target_column=target_column),
    }
