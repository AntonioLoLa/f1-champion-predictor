"""Tests for walk-forward cross-validation."""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.training.splitters import (
    Fold,
    RollingWindowSplit,
    SeasonForwardSplit,
    completed_seasons,
)

pytestmark = pytest.mark.unit


def _frame(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """Build a minimal ``season``/``round``/``rounds_total`` frame."""
    return pl.DataFrame(
        rows,
        schema=["season", "round", "rounds_total"],
        orient="row",
    )


def test_completed_seasons_excludes_a_season_in_progress() -> None:
    frame = _frame(
        [
            (2020, 1, 2),
            (2020, 2, 2),
            (2021, 1, 3),  # only round 1 of 3 stored: still in progress
        ]
    )
    assert completed_seasons(frame) == [2020]


def test_completed_seasons_is_ascending() -> None:
    frame = _frame([(2022, 2, 2), (2020, 1, 1), (2021, 1, 1)])
    assert completed_seasons(frame) == [2020, 2021, 2022]


def test_fold_rejects_overlapping_seasons() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        Fold(train_seasons=[2020, 2021], test_seasons=[2021, 2022])


def test_fold_rejects_test_seasons_before_train_seasons() -> None:
    with pytest.raises(ValueError, match="must all come after"):
        Fold(train_seasons=[2020, 2022], test_seasons=[2021])


def test_fold_accepts_a_valid_split() -> None:
    fold = Fold(train_seasons=[2020, 2021], test_seasons=[2022])
    assert fold.train_seasons == [2020, 2021]
    assert fold.test_seasons == [2022]


def test_season_forward_split_expands_the_training_window() -> None:
    seasons = list(range(2000, 2006))  # 2000..2005, six seasons
    splitter = SeasonForwardSplit(min_train_seasons=3, test_horizon_seasons=1)
    folds = list(splitter.split(seasons))

    assert [fold.train_seasons for fold in folds] == [
        [2000, 2001, 2002],
        [2000, 2001, 2002, 2003],
        [2000, 2001, 2002, 2003, 2004],
    ]
    assert [fold.test_seasons for fold in folds] == [[2003], [2004], [2005]]


def test_season_forward_split_yields_nothing_below_the_minimum() -> None:
    splitter = SeasonForwardSplit(min_train_seasons=10, test_horizon_seasons=1)
    assert list(splitter.split([2020, 2021, 2022])) == []


def test_season_forward_split_respects_the_test_horizon() -> None:
    seasons = list(range(2000, 2006))
    splitter = SeasonForwardSplit(min_train_seasons=3, test_horizon_seasons=2)
    folds = list(splitter.split(seasons))
    assert [fold.test_seasons for fold in folds] == [[2003, 2004]]


def test_rolling_window_split_keeps_a_fixed_width_window() -> None:
    seasons = list(range(2000, 2007))  # seven seasons
    splitter = RollingWindowSplit(window_seasons=3, test_horizon_seasons=1)
    folds = list(splitter.split(seasons))

    assert [fold.train_seasons for fold in folds] == [
        [2000, 2001, 2002],
        [2001, 2002, 2003],
        [2002, 2003, 2004],
        [2003, 2004, 2005],
    ]
    assert [fold.test_seasons for fold in folds] == [[2003], [2004], [2005], [2006]]


def test_rolling_window_split_yields_nothing_below_the_window_size() -> None:
    splitter = RollingWindowSplit(window_seasons=10, test_horizon_seasons=1)
    assert list(splitter.split([2020, 2021, 2022])) == []
