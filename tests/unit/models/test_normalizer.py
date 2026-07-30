"""Tests for per-group probability normalisation."""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.models.normalizer import GroupProbabilityNormalizer

pytestmark = pytest.mark.unit


def test_normalize_rescales_probabilities_to_sum_to_one_per_group() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1, 1],
            "probability": [0.2, 0.2, 0.2, 0.5, 0.5],
        }
    )
    normalizer = GroupProbabilityNormalizer(["season", "round"])
    result = normalizer.normalize(frame, "probability")

    sums = result.group_by(["season", "round"]).agg(pl.col("probability_normalized").sum())
    assert all(value == pytest.approx(1.0) for value in sums["probability_normalized"])


def test_normalize_preserves_relative_ordering_within_a_group() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2020],
            "round": [1, 1, 1],
            "probability": [0.1, 0.6, 0.3],
        }
    )
    result = GroupProbabilityNormalizer(["season", "round"]).normalize(frame, "probability")
    normalized = result["probability_normalized"].to_list()
    assert normalized[1] > normalized[2] > normalized[0]


def test_normalize_falls_back_to_equal_share_when_the_group_sums_to_zero() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "probability": [0.0, 0.0],
        }
    )
    result = GroupProbabilityNormalizer(["season", "round"]).normalize(frame, "probability")
    assert result["probability_normalized"].to_list() == [0.5, 0.5]


def test_normalize_handles_multiple_independent_groups() -> None:
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "round": [1, 1, 1, 1],
            "probability": [0.4, 0.4, 1.0, 0.0],
        }
    )
    result = GroupProbabilityNormalizer(["season", "round"]).normalize(frame, "probability")
    assert result["probability_normalized"].to_list() == pytest.approx([0.5, 0.5, 1.0, 0.0])
