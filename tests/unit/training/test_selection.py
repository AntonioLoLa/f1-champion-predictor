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
    fold_metrics: list[dict[str, float | None]] = field(default_factory=list)


def _folds(*values: float, metric: str = "group_log_loss") -> list[dict[str, float | None]]:
    return [{metric: value} for value in values]


def test_select_best_picks_the_lowest_primary_metric_when_minimizing() -> None:
    config = SelectionConfig(primary_metric="group_log_loss", direction="minimize")
    candidates = [
        _Candidate("a", {"group_log_loss": 0.5}, _folds(0.4, 0.5, 0.6)),
        _Candidate("b", {"group_log_loss": 0.2}, _folds(0.1, 0.2, 0.3)),
        _Candidate("c", {"group_log_loss": 0.9}, _folds(0.8, 0.9, 1.0)),
    ]
    result = select_best(candidates, config)
    assert result.winner.name == "b"
    assert result.tied == ["b"]


def test_select_best_picks_the_highest_primary_metric_when_maximizing() -> None:
    config = SelectionConfig(primary_metric="top1_hit_rate", direction="maximize")
    candidates = [
        _Candidate("a", {"top1_hit_rate": 0.5}, _folds(0.4, 0.5, 0.6, metric="top1_hit_rate")),
        _Candidate("b", {"top1_hit_rate": 0.9}, _folds(0.8, 0.9, 1.0, metric="top1_hit_rate")),
    ]
    assert select_best(candidates, config).winner.name == "b"


def test_select_best_breaks_ties_with_the_configured_tie_breakers() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        tie_breakers=["brier_score", "expected_calibration_error"],
    )
    candidates = [
        _Candidate("a", {"group_log_loss": 0.5, "brier_score": 0.3}, _folds(0.5, 0.5, 0.5)),
        _Candidate("b", {"group_log_loss": 0.5, "brier_score": 0.1}, _folds(0.5, 0.5, 0.5)),
    ]
    result = select_best(candidates, config)
    assert result.winner.name == "b"
    assert set(result.tied) == {"a", "b"}


def test_select_best_falls_through_to_the_second_tie_breaker() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        tie_breakers=["brier_score", "expected_calibration_error"],
    )
    candidates = [
        _Candidate(
            "a",
            {"group_log_loss": 0.5, "brier_score": 0.2, "expected_calibration_error": 0.1},
            _folds(0.5, 0.5, 0.5),
        ),
        _Candidate(
            "b",
            {"group_log_loss": 0.5, "brier_score": 0.2, "expected_calibration_error": 0.05},
            _folds(0.5, 0.5, 0.5),
        ),
    ]
    assert select_best(candidates, config).winner.name == "b"


def test_select_best_treats_a_missing_metric_as_worst() -> None:
    config = SelectionConfig(primary_metric="group_log_loss", direction="minimize")
    candidates = [
        _Candidate("a", {"group_log_loss": None}),
        _Candidate("b", {"group_log_loss": 0.5}, _folds(0.4, 0.5, 0.6)),
    ]
    assert select_best(candidates, config).winner.name == "b"


def test_select_best_rejects_an_empty_candidate_list() -> None:
    config = SelectionConfig()
    with pytest.raises(ValueError, match="empty"):
        select_best([], config)


def test_select_best_applies_the_one_standard_error_rule() -> None:
    """A runner-up within one SE of the leader is not a settled win -- prefer simplicity."""
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        one_standard_error=True,
        preference_order=["simple", "complex"],
    )
    candidates = [
        # Mean 0.60, but folds swing widely: SE is large enough to swallow "complex".
        _Candidate("simple", {"group_log_loss": 0.60}, _folds(0.1, 0.6, 1.1)),
        _Candidate("complex", {"group_log_loss": 0.62}, _folds(0.5, 0.6, 0.66)),
    ]
    result = select_best(candidates, config)
    assert result.rule == "one_standard_error"
    assert result.winner.name == "simple"
    assert set(result.tied) == {"simple", "complex"}
    assert result.standard_error is not None


def test_select_best_does_not_widen_the_tie_when_the_gap_is_real() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss",
        direction="minimize",
        one_standard_error=True,
        preference_order=["worse"],
    )
    candidates = [
        _Candidate("best", {"group_log_loss": 0.2}, _folds(0.15, 0.2, 0.25)),
        _Candidate("worse", {"group_log_loss": 5.0}, _folds(4.9, 5.0, 5.1)),
    ]
    result = select_best(candidates, config)
    assert result.winner.name == "best"
    assert result.tied == ["best"]


def test_select_best_falls_back_to_the_bare_mean_when_the_rule_is_disabled() -> None:
    config = SelectionConfig(
        primary_metric="group_log_loss", direction="minimize", one_standard_error=False
    )
    candidates = [
        _Candidate("a", {"group_log_loss": 0.60}, _folds(0.1, 0.6, 1.1)),
        _Candidate("b", {"group_log_loss": 0.62}, _folds(0.5, 0.6, 0.66)),
    ]
    result = select_best(candidates, config)
    assert result.rule == "best_mean"
    assert result.winner.name == "a"
