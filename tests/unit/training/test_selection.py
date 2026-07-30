"""Tests for model selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from f1cp.config.settings import SelectionConfig
from f1cp.training.selection import select_best

pytestmark = pytest.mark.unit


@dataclass(slots=True)
class _Candidate:
    """A minimal `Scored` implementation for testing selection in isolation."""

    name: str
    metrics: dict[str, float | None] = field(default_factory=dict)


def test_select_best_picks_the_lowest_primary_metric_when_minimizing() -> None:
    config = SelectionConfig(primary_metric="group_log_loss", direction="minimize")
    candidates = [
        _Candidate("a", {"group_log_loss": 0.5}),
        _Candidate("b", {"group_log_loss": 0.2}),
        _Candidate("c", {"group_log_loss": 0.9}),
    ]
    assert select_best(candidates, config).name == "b"


def test_select_best_picks_the_highest_primary_metric_when_maximizing() -> None:
    config = SelectionConfig(primary_metric="top1_hit_rate", direction="maximize")
    candidates = [
        _Candidate("a", {"top1_hit_rate": 0.5}),
        _Candidate("b", {"top1_hit_rate": 0.9}),
    ]
    assert select_best(candidates, config).name == "b"


def test_select_best_breaks_ties_with_the_configured_tie_breakers() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        tie_breakers=["brier_score", "expected_calibration_error"],
    )
    candidates = [
        _Candidate("a", {"group_log_loss": 0.5, "brier_score": 0.3}),
        _Candidate("b", {"group_log_loss": 0.5, "brier_score": 0.1}),
    ]
    assert select_best(candidates, config).name == "b"


def test_select_best_falls_through_to_the_second_tie_breaker() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        tie_breakers=["brier_score", "expected_calibration_error"],
    )
    candidates = [
        _Candidate(
            "a", {"group_log_loss": 0.5, "brier_score": 0.2, "expected_calibration_error": 0.1}
        ),
        _Candidate(
            "b", {"group_log_loss": 0.5, "brier_score": 0.2, "expected_calibration_error": 0.05}
        ),
    ]
    assert select_best(candidates, config).name == "b"


def test_select_best_treats_a_missing_metric_as_worst() -> None:
    config = SelectionConfig(primary_metric="group_log_loss", direction="minimize")
    candidates = [
        _Candidate("a", {"group_log_loss": None}),
        _Candidate("b", {"group_log_loss": 0.5}),
    ]
    assert select_best(candidates, config).name == "b"


def test_select_best_rejects_an_empty_candidate_list() -> None:
    config = SelectionConfig()
    with pytest.raises(ValueError, match="empty"):
        select_best([], config)
