"""The fuller evaluation metric suite, Plotly reporting figures, and report generation."""

from f1cp.evaluation.metrics import calibration_curve_points, compute_full_metrics
from f1cp.evaluation.plots import (
    calibration_curve_figure,
    confusion_matrix_figure,
    metric_by_phase_figure,
    metric_by_season_figure,
    pr_curve_figure,
    roc_curve_figure,
)
from f1cp.evaluation.report import EvaluationReport, build_report

__all__ = [
    "EvaluationReport",
    "build_report",
    "calibration_curve_figure",
    "calibration_curve_points",
    "compute_full_metrics",
    "confusion_matrix_figure",
    "metric_by_phase_figure",
    "metric_by_season_figure",
    "pr_curve_figure",
    "roc_curve_figure",
]
