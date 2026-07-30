"""Walk-forward validation, scoring, selection, MLflow tracking and training orchestration."""

from f1cp.training.metrics import (
    brier_score,
    compute_metrics,
    expected_calibration_error,
    final_round_top1_hit_rate,
    group_log_loss,
    top1_hit_rate,
    top_ranked_mask,
)
from f1cp.training.selection import Scored, select_best
from f1cp.training.splitters import (
    Fold,
    RollingWindowSplit,
    SeasonForwardSplit,
    Splitter,
    completed_seasons,
)
from f1cp.training.tracking import ExperimentTracker
from f1cp.training.trainer import CandidateResult, Trainer, TrainingResult

__all__ = [
    "CandidateResult",
    "ExperimentTracker",
    "Fold",
    "RollingWindowSplit",
    "Scored",
    "SeasonForwardSplit",
    "Splitter",
    "Trainer",
    "TrainingResult",
    "brier_score",
    "completed_seasons",
    "compute_metrics",
    "expected_calibration_error",
    "final_round_top1_hit_rate",
    "group_log_loss",
    "select_best",
    "top1_hit_rate",
    "top_ranked_mask",
]
