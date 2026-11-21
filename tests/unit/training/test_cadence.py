"""Tests for the rule that decides whether a scheduled retrain is worth running."""

from __future__ import annotations

import datetime as dt

import pytest

from f1cp.training.cadence import is_due

pytestmark = pytest.mark.unit

_FINAL_RACE = dt.date(2026, 12, 6)


def _at(day: dt.date, hour: int = 12) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.UTC)


def test_not_due_mid_season() -> None:
    """2025 closed in December 2025; a run in August 2026 already saw it."""
    assert not is_due(
        season_closed_on=dt.date(2025, 12, 7), last_trained_at=_at(dt.date(2026, 8, 3))
    )


def test_due_once_the_season_closes() -> None:
    assert is_due(season_closed_on=_FINAL_RACE, last_trained_at=_at(dt.date(2026, 8, 3)))


def test_a_run_on_the_day_of_the_final_counts_as_before_it() -> None:
    """Results may land after a same-day run, so that run cannot be trusted to include them."""
    assert is_due(season_closed_on=_FINAL_RACE, last_trained_at=_at(_FINAL_RACE, hour=23))


def test_not_due_after_retraining() -> None:
    assert not is_due(
        season_closed_on=_FINAL_RACE, last_trained_at=_at(_FINAL_RACE + dt.timedelta(days=1))
    )


def test_due_when_nothing_was_ever_trained() -> None:
    assert is_due(season_closed_on=_FINAL_RACE, last_trained_at=None)


def test_never_due_without_a_complete_season() -> None:
    assert not is_due(season_closed_on=None, last_trained_at=None)
