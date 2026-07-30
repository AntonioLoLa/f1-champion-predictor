"""Tests for walk-forward CV scoring."""

from __future__ import annotations

import math

import polars as pl
import pytest

from f1cp.training.metrics import (
    brier_score,
    compute_metrics,
    expected_calibration_error,
    final_round_top1_hit_rate,
    group_log_loss,
    top1_hit_rate,
    top_ranked_mask,
)

pytestmark = pytest.mark.unit


def test_group_log_loss_scores_the_true_positive_row() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1],
            "champion": [1, 0, 0, 1],
            "probability": [0.8, 0.2, 0.4, 0.6],
        }
    )
    loss = group_log_loss(
        frame,
        probability_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    expected = -(math.log(0.8) + math.log(0.6)) / 2
    assert loss is not None
    assert loss == pytest.approx(expected)


def test_group_log_loss_returns_none_without_a_positive_row() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020],
            "round": [1],
            "champion": [0],
            "probability": [0.5],
        }
    )
    assert (
        group_log_loss(
            frame,
            probability_column="probability",
            target_column="champion",
            group_columns=["season", "round"],
        )
        is None
    )


def test_group_log_loss_rejects_a_group_with_two_positives() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "champion": [1, 1],
            "probability": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="more than one positive"):
        group_log_loss(
            frame,
            probability_column="probability",
            target_column="champion",
            group_columns=["season", "round"],
        )


def test_brier_score_is_zero_for_a_perfect_prediction() -> None:
    frame = pl.DataFrame({"champion": [1, 0], "probability": [1.0, 0.0]})
    assert brier_score(frame, probability_column="probability", target_column="champion") == 0.0


def test_brier_score_matches_hand_computation() -> None:
    frame = pl.DataFrame({"champion": [1, 0], "probability": [0.7, 0.3]})
    score = brier_score(frame, probability_column="probability", target_column="champion")
    expected = ((0.7 - 1) ** 2 + (0.3 - 0) ** 2) / 2
    assert score == pytest.approx(expected)


def test_brier_score_returns_none_for_an_empty_frame() -> None:
    frame = pl.DataFrame({"champion": [], "probability": []})
    assert brier_score(frame, probability_column="probability", target_column="champion") is None


def test_expected_calibration_error_is_zero_when_perfectly_calibrated() -> None:
    # Every predicted probability equals the observed rate within its bin.
    frame = pl.DataFrame(
        {
            "champion": [1, 0, 1, 0],
            "probability": [0.5, 0.5, 0.5, 0.5],
        }
    )
    ece = expected_calibration_error(
        frame, probability_column="probability", target_column="champion", n_bins=10
    )
    assert ece == pytest.approx(0.0)


def test_expected_calibration_error_penalises_overconfidence() -> None:
    frame = pl.DataFrame({"champion": [0, 0], "probability": [0.9, 0.9]})
    ece = expected_calibration_error(
        frame, probability_column="probability", target_column="champion", n_bins=10
    )
    assert ece == pytest.approx(0.9)


def test_top1_hit_rate_counts_the_highest_ranked_row_per_group() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1],
            "champion": [1, 0, 0, 1],
            "probability": [0.9, 0.1, 0.9, 0.1],
        }
    )
    rate = top1_hit_rate(
        frame,
        probability_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    # 2020 correctly ranks the champion first; 2021 does not.
    assert rate == pytest.approx(0.5)


def test_top1_hit_rate_returns_none_for_an_empty_frame() -> None:
    frame = pl.DataFrame({"season": [], "round": [], "champion": [], "probability": []})
    assert (
        top1_hit_rate(
            frame,
            probability_column="probability",
            target_column="champion",
            group_columns=["season", "round"],
        )
        is None
    )


def test_final_round_top1_hit_rate_only_looks_at_each_seasons_last_round() -> None:
    # Two drivers per round; the label is constant for a driver across their whole season.
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
            "round": [1, 1, 2, 2, 1, 1, 2, 2],
            "champion": [1, 0, 1, 0, 0, 1, 0, 1],
            "probability": [0.4, 0.6, 0.9, 0.1, 0.9, 0.1, 0.8, 0.2],
        }
    )
    rate = final_round_top1_hit_rate(
        frame, probability_column="probability", target_column="champion"
    )
    # 2020's final round (round 2) correctly ranks its champion first (0.9 vs 0.1);
    # 2021's final round does not (champion gets 0.2, runner-up gets 0.8).
    assert rate == pytest.approx(0.5)


def test_compute_metrics_bundles_every_metric() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "champion": [1, 0],
            "probability": [0.8, 0.2],
        }
    )
    result = compute_metrics(
        frame,
        calibrated_column="probability",
        normalized_column="probability",
        target_column="champion",
        group_columns=["season", "round"],
    )
    assert set(result) == {
        "group_log_loss",
        "brier_score",
        "expected_calibration_error",
        "top1_hit_rate",
    }
    assert all(value is not None for value in result.values())


def test_top_ranked_mask_flags_the_highest_probability_row_per_group() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1],
            "probability": [0.9, 0.1, 0.3, 0.7],
        }
    )
    mask = top_ranked_mask(
        frame, probability_column="probability", group_columns=["season", "round"]
    )
    assert mask.to_list() == [True, False, False, True]
