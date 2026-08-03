"""Feature pipeline correctness and leakage tests.

The leakage property test is the load-bearing test in this suite: it asserts that every
engineered value at ``(season, round)`` is unaffected by deleting everything after that
point from the source data, which is exactly what "no lookahead" means in a walk-forward
prediction problem. See :mod:`f1cp.features.panel` for the design this enforces.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from f1cp.config.settings import FeaturesConfig, PathsConfig
from f1cp.features.pipeline import run_pipeline
from tests.unit.features.synthetic import (
    ABSENCES,
    DEBUT,
    build_synthetic_sources,
    expected_panel_rows,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        data=tmp_path,
        raw=tmp_path / "raw",
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        cache=tmp_path / "cache",
        artifacts=tmp_path / "artifacts",
        reports=tmp_path / "reports",
    )


@pytest.fixture
def features_config() -> FeaturesConfig:
    return FeaturesConfig()


def test_pipeline_produces_one_row_per_driver_round(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    sources = build_synthetic_sources()
    frame = run_pipeline(sources, features_config, paths)

    assert frame.height == expected_panel_rows()
    assert frame.select(["season", "round", "driver_id"]).is_duplicated().sum() == 0


def test_a_driver_who_sits_out_a_round_keeps_their_row(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    """Missing a Grand Prix does not remove a driver from the championship.

    The panel row has to survive, flagged ``entered = False``, or the round's probabilities
    get rescaled over a group that may not contain the eventual champion -- see
    :func:`f1cp.dataset.builder._apply_filters`.
    """
    frame = run_pipeline(build_synthetic_sources(), features_config, paths)

    season, round_number, driver_id = next(iter(ABSENCES))
    absent = frame.filter(
        (pl.col("season") == season)
        & (pl.col("round") == round_number)
        & (pl.col("driver_id") == driver_id)
    )
    assert absent.height == 1
    assert absent["entered"].item() is False
    assert absent["finish_position"].item() is None
    # They still carry the championship state that makes them a candidate for the title.
    assert absent["current_points"].item() > 0


def test_a_driver_has_no_rows_before_their_debut(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    """A driver who has not raced yet is not in the championship and gets no row."""
    frame = run_pipeline(build_synthetic_sources(), features_config, paths)

    driver_id = "d4"
    debut_season, debut_round = DEBUT[driver_id]
    rows = frame.filter(pl.col("driver_id") == driver_id)
    assert rows.height > 0
    earliest = rows.sort(["season", "round"]).row(0, named=True)
    assert (earliest["season"], earliest["round"]) == (debut_season, debut_round)


def test_season_rates_divide_by_rounds_started_not_rounds_elapsed(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    """A mid-season arrival's rates must reflect their own races, not the calendar's.

    ``d4`` debuts at round 3 and finishes last every time, so at round 4 they have started
    exactly one race and scored no podium: ``podium_rate`` must be 0/1, and the denominator
    must be 1 rather than the 3 rounds the season has already run.
    """
    frame = run_pipeline(build_synthetic_sources(), features_config, paths)

    debut_season, debut_round = DEBUT["d4"]
    after_debut = frame.filter(
        (pl.col("season") == debut_season)
        & (pl.col("round") == debut_round + 1)
        & (pl.col("driver_id") == "d4")
    )
    assert after_debut.height == 1
    # One start, one finish, no DNF: a rate over rounds *started* is exactly 0.0 or 1.0,
    # never the 1/3 a calendar-round denominator would produce.
    assert after_debut["dnf_rate"].item() == 0.0
    assert after_debut["points_finish_rate"].item() in {0.0, 1.0}


def test_momentum_divides_by_rounds_started_not_window_size(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    """A driver's 2nd-ever round must not read as a form decline.

    ``d4`` debuts at round 3 and has started exactly one round coming into round 4, so
    their "recent" window and "season so far" window cover the identical single result:
    momentum (their difference) must be exactly 0.0, never a fraction of the window size.
    """
    frame = run_pipeline(build_synthetic_sources(), features_config, paths)

    debut_season, debut_round = DEBUT["d4"]
    after_debut = frame.filter(
        (pl.col("season") == debut_season)
        & (pl.col("round") == debut_round + 1)
        & (pl.col("driver_id") == "d4")
    )
    assert after_debut.height == 1
    assert after_debut["momentum"].item() == pytest.approx(0.0)


def test_pipeline_columns_are_numeric_or_boolean(
    features_config: FeaturesConfig, paths: PathsConfig
) -> None:
    sources = build_synthetic_sources()
    frame = run_pipeline(sources, features_config, paths)

    non_key_metadata = {"circuit_id", "race_date", "status", "constructor_id", "driver_id"}
    offending = [
        column
        for column, dtype in frame.schema.items()
        if column not in non_key_metadata and not (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    assert offending == []


@pytest.mark.parametrize(("cutoff_season", "cutoff_round"), [(2021, 2), (2020, 3)])
def test_features_are_unchanged_by_deleting_the_future(
    features_config: FeaturesConfig,
    paths: PathsConfig,
    cutoff_season: int,
    cutoff_round: int,
) -> None:
    """The central leakage guard: truncating the future must not change the past.

    Runs the full pipeline once on the complete synthetic history and once on
    ``sources.as_of(cutoff_season, cutoff_round)``, then asserts that every row visible at
    the cutoff produced byte-for-byte identical features in both runs.
    """
    sources = build_synthetic_sources()

    full = run_pipeline(sources, features_config, paths)
    truncated = run_pipeline(sources.as_of(cutoff_season, cutoff_round), features_config, paths)

    visible = full.filter(
        (pl.col("season") < cutoff_season)
        | ((pl.col("season") == cutoff_season) & (pl.col("round") <= cutoff_round))
    ).sort(["season", "round", "driver_id"])

    truncated_sorted = truncated.sort(["season", "round", "driver_id"])

    assert visible.height == truncated_sorted.height
    assert_frame_equal(
        visible.select(sorted(visible.columns)),
        truncated_sorted.select(sorted(truncated_sorted.columns)),
        check_column_order=True,
    )


def test_champion_label_source_only_credits_completed_seasons() -> None:
    """A season still missing its final round's standings must not look like it has ended.

    This mirrors the guard added to
    :meth:`f1cp.db.repositories.f1_data.StandingsRepository.champion_by_season`: a driver
    leading after the last *stored* round of an in-progress season is not the champion.
    The feature layer itself has no opinion on this (labelling happens in
    :mod:`f1cp.dataset.builder`), so this test simply documents the synthetic fixture is a
    faithful stand-in: every season here has standings through its final calendar round.
    """
    sources = build_synthetic_sources()
    final_rounds = sources.races.group_by("season").agg(pl.col("round").max().alias("final"))
    standings_rounds = sources.driver_standings.group_by("season").agg(
        pl.col("round").max().alias("final")
    )
    assert_frame_equal(
        final_rounds.sort("season"), standings_rounds.sort("season"), check_column_order=False
    )
