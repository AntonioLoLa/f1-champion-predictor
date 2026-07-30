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
from tests.unit.features.synthetic import ROUNDS, SEASONS, build_synthetic_sources

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

    # 2 seasons x 4 rounds x 3 drivers, every driver contests every round.
    assert frame.height == len(SEASONS) * len(ROUNDS) * 3
    assert frame.select(["season", "round", "driver_id"]).is_duplicated().sum() == 0


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
