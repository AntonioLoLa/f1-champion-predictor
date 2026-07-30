"""Tests for the fuller evaluation metric suite."""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.evaluation.metrics import calibration_curve_points, compute_full_metrics

pytestmark = pytest.mark.unit


def _frame() -> pl.DataFrame:
    """Two groups, each with a clear top-ranked row matching the true champion."""
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1],
            "champion": [1, 0, 0, 1],
            "probability": [0.9, 0.1, 0.2, 0.8],
        }
    )


def test_compute_full_metrics_scores_a_perfectly_ranked_frame() -> None:
    result = compute_full_metrics(
        _frame(),
        calibrated_column="probability",
        normalized_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["log_loss"] is not None


def test_compute_full_metrics_penalises_a_wrong_ranking() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "champion": [1, 0],
            "probability": [0.1, 0.9],
        }
    )
    result = compute_full_metrics(
        frame,
        calibrated_column="probability",
        normalized_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["true_positive"] == 0
    assert result["false_negative"] == 1
    assert result["false_positive"] == 1


def test_compute_full_metrics_handles_an_empty_frame() -> None:
    frame = pl.DataFrame(
        {"season": [], "round": [], "champion": [], "probability": []},
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "champion": pl.Int64,
            "probability": pl.Float64,
        },
    )
    result = compute_full_metrics(
        frame,
        calibrated_column="probability",
        normalized_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    assert all(value is None for value in result.values())


def test_compute_full_metrics_handles_a_single_class_slice() -> None:
    # A phase subset where nobody present is the eventual champion: ROC/PR AUC/log loss
    # are undefined, but the classification metrics (all-negative predictions) still are.
    frame = pl.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "champion": [0, 0],
            "probability": [0.4, 0.6],
        }
    )
    result = compute_full_metrics(
        frame,
        calibrated_column="probability",
        normalized_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    assert result["roc_auc"] is None
    assert result["pr_auc"] is None
    assert result["log_loss"] is None
    # The higher-probability (but still non-champion) row gets picked as the group's
    # top-1: one right (the other, correctly unpicked, row), one wrong.
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1


def test_calibration_curve_points_matches_predicted_confidence_when_well_calibrated() -> None:
    frame = pl.DataFrame(
        {"champion": [1, 0, 1, 0], "probability": [0.5, 0.5, 0.5, 0.5]},
    )
    points = calibration_curve_points(
        frame, calibrated_column="probability", target_column="champion", n_bins=10
    )
    assert len(points) == 1
    assert points[0]["confidence"] == pytest.approx(0.5)
    assert points[0]["accuracy"] == pytest.approx(0.5)
    assert points[0]["count"] == 4


def test_calibration_curve_points_is_empty_for_an_empty_frame() -> None:
    frame = pl.DataFrame(
        {"champion": [], "probability": []},
        schema={"champion": pl.Int64, "probability": pl.Float64},
    )
    assert (
        calibration_curve_points(frame, calibrated_column="probability", target_column="champion")
        == []
    )
