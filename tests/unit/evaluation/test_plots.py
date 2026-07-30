"""Tests for evaluation Plotly figures."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import pytest

from f1cp.evaluation.plots import (
    calibration_curve_figure,
    confusion_matrix_figure,
    metric_by_phase_figure,
    metric_by_season_figure,
    pr_curve_figure,
    roc_curve_figure,
)

pytestmark = pytest.mark.unit


def test_calibration_curve_figure_includes_the_diagonal_and_observed_points() -> None:
    points = [
        {"bin_center": 0.25, "confidence": 0.3, "accuracy": 0.2, "count": 10},
        {"bin_center": 0.75, "confidence": 0.7, "accuracy": 0.8, "count": 5},
    ]
    figure = calibration_curve_figure(points)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.data[1].x == (0.3, 0.7)


def test_calibration_curve_figure_handles_no_points() -> None:
    figure = calibration_curve_figure([])
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1  # just the diagonal


def test_confusion_matrix_figure_lays_out_a_2x2_heatmap() -> None:
    counts = {
        "true_positive": 3.0,
        "false_positive": 1.0,
        "false_negative": 2.0,
        "true_negative": 4.0,
    }
    figure = confusion_matrix_figure(counts)
    assert isinstance(figure.data[0], go.Heatmap)
    assert list(figure.data[0].z) == [[3.0, 2.0], [1.0, 4.0]]


def test_confusion_matrix_figure_handles_missing_values() -> None:
    figure = confusion_matrix_figure({})
    assert list(figure.data[0].z) == [[0, 0], [0, 0]]


def test_roc_curve_figure_draws_a_curve_for_two_classes() -> None:
    frame = pl.DataFrame({"champion": [1, 0, 1, 0], "probability": [0.9, 0.1, 0.6, 0.4]})
    figure = roc_curve_figure(frame, calibrated_column="probability", target_column="champion")
    assert len(figure.data) == 2  # chance line + ROC


def test_roc_curve_figure_omits_the_curve_for_a_single_class() -> None:
    frame = pl.DataFrame({"champion": [0, 0], "probability": [0.1, 0.2]})
    figure = roc_curve_figure(frame, calibrated_column="probability", target_column="champion")
    assert len(figure.data) == 1  # chance line only


def test_pr_curve_figure_draws_a_curve_for_two_classes() -> None:
    frame = pl.DataFrame({"champion": [1, 0, 1, 0], "probability": [0.9, 0.1, 0.6, 0.4]})
    figure = pr_curve_figure(frame, calibrated_column="probability", target_column="champion")
    assert len(figure.data) == 1


def test_metric_by_phase_figure_orders_bars_early_mid_late() -> None:
    phase_metrics = {
        "late": {"top1_hit_rate": 0.9},
        "early": {"top1_hit_rate": 0.4},
        "mid": {"top1_hit_rate": 0.6},
    }
    figure = metric_by_phase_figure(phase_metrics, metric_name="top1_hit_rate")
    assert figure.data[0].x == ("early", "mid", "late")
    assert figure.data[0].y == (0.4, 0.6, 0.9)


def test_metric_by_season_figure_orders_by_season_ascending() -> None:
    season_metrics = {
        2022: {"top1_hit_rate": 0.9},
        2020: {"top1_hit_rate": 0.5},
        2021: {"top1_hit_rate": 0.7},
    }
    figure = metric_by_season_figure(season_metrics, metric_name="top1_hit_rate")
    assert figure.data[0].x == (2020, 2021, 2022)
    assert figure.data[0].y == (0.5, 0.7, 0.9)
