"""Plotly figures for the evaluation report.

Every function returns a bare ``plotly.graph_objects.Figure`` and takes data the caller
already has in hand (a scored frame, or metrics :mod:`f1cp.evaluation.report` already
computed) -- this module only draws, it never recomputes a metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

__all__ = [
    "calibration_curve_figure",
    "confusion_matrix_figure",
    "metric_by_phase_figure",
    "metric_by_season_figure",
    "pr_curve_figure",
    "roc_curve_figure",
]

_DEFAULT_PHASE_ORDER = ("early", "mid", "late")


def calibration_curve_figure(points: list[dict[str, float]]) -> go.Figure:
    """Reliability diagram: predicted confidence vs. observed accuracy, per bin.

    Args:
        points: Output of :func:`f1cp.evaluation.metrics.calibration_curve_points`.

    Returns:
        A scatter of observed-vs-predicted points against the perfect-calibration
        diagonal, marker size proportional to bin count.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash"}
        )
    )
    if points:
        figure.add_trace(
            go.Scatter(
                x=[p["confidence"] for p in points],
                y=[p["accuracy"] for p in points],
                mode="markers+lines",
                name="Observed",
                marker={
                    "size": [
                        8 + 20 * (p["count"] / max(pt["count"] for pt in points)) for p in points
                    ]
                },
            )
        )
    figure.update_layout(
        title="Calibration curve",
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return figure


def confusion_matrix_figure(counts: Mapping[str, float | None]) -> go.Figure:
    """Heatmap of the 2x2 confusion matrix for the per-group top-1 prediction.

    Args:
        counts: A mapping with ``true_positive``, ``false_positive``, ``false_negative``
            and ``true_negative`` (as produced by
            :func:`f1cp.evaluation.metrics.compute_full_metrics`).

    Returns:
        A heatmap, rows/columns labelled "champion"/"not champion".
    """
    tp = counts.get("true_positive") or 0
    fp = counts.get("false_positive") or 0
    fn = counts.get("false_negative") or 0
    tn = counts.get("true_negative") or 0
    z = [[tp, fn], [fp, tn]]
    labels = ["Predicted champion", "Predicted not champion"]
    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Actual champion", "Actual not champion"],
            y=labels,
            text=z,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    figure.update_layout(title="Confusion matrix (per-group top-1 prediction)")
    return figure


def roc_curve_figure(
    frame: pl.DataFrame, *, calibrated_column: str, target_column: str
) -> go.Figure:
    """ROC curve for the calibrated, pre-normalisation probability.

    Args:
        frame: Rows carrying ``calibrated_column`` and ``target_column``.
        calibrated_column: Pointwise calibrated probability column.
        target_column: Binary champion label.

    Returns:
        The ROC curve against the random-guess diagonal. Empty (axes only) if ``frame``
        has fewer than two classes.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"})
    )
    y_true = frame[target_column].to_numpy() if not frame.is_empty() else np.array([])
    if len(np.unique(y_true)) >= 2:
        y_score = frame[calibrated_column].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    figure.update_layout(
        title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate"
    )
    return figure


def pr_curve_figure(
    frame: pl.DataFrame, *, calibrated_column: str, target_column: str
) -> go.Figure:
    """Precision-recall curve for the calibrated, pre-normalisation probability.

    Args:
        frame: Rows carrying ``calibrated_column`` and ``target_column``.
        calibrated_column: Pointwise calibrated probability column.
        target_column: Binary champion label.

    Returns:
        The precision-recall curve. Empty (axes only) if ``frame`` has fewer than two
        classes.
    """
    figure = go.Figure()
    y_true = frame[target_column].to_numpy() if not frame.is_empty() else np.array([])
    if len(np.unique(y_true)) >= 2:
        y_score = frame[calibrated_column].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    figure.update_layout(
        title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision"
    )
    return figure


def metric_by_phase_figure(
    phase_metrics: Mapping[str, Mapping[str, float | None]],
    *,
    metric_name: str,
    phase_order: Sequence[str] = _DEFAULT_PHASE_ORDER,
) -> go.Figure:
    """Bar chart of one metric across season phases -- the project's headline chart.

    Args:
        phase_metrics: Mapping of phase name to its metrics dict.
        metric_name: Which metric to plot.
        phase_order: Display order for the phases present in ``phase_metrics``.

    Returns:
        A bar chart, one bar per phase present in both ``phase_order`` and
        ``phase_metrics``.
    """
    phases = [phase for phase in phase_order if phase in phase_metrics]
    values = [phase_metrics[phase].get(metric_name) for phase in phases]
    figure = go.Figure(data=go.Bar(x=phases, y=values))
    figure.update_layout(
        title=f"{metric_name} by season phase", xaxis_title="Phase", yaxis_title=metric_name
    )
    return figure


def metric_by_season_figure(
    season_metrics: Mapping[int, Mapping[str, float | None]],
    *,
    metric_name: str,
) -> go.Figure:
    """Line chart of one metric across seasons, in chronological order.

    Args:
        season_metrics: Mapping of season to its metrics dict.
        metric_name: Which metric to plot.

    Returns:
        A line+marker chart over the seasons present in ``season_metrics``, ascending.
    """
    seasons = sorted(season_metrics)
    values = [season_metrics[season].get(metric_name) for season in seasons]
    figure = go.Figure(data=go.Scatter(x=seasons, y=values, mode="lines+markers"))
    figure.update_layout(
        title=f"{metric_name} by season", xaxis_title="Season", yaxis_title=metric_name
    )
    return figure
