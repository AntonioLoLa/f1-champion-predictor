"""``CircuitBlock`` values, checked against a slow row-at-a-time reference.

The leakage test cannot catch this block's historical defect, and docs/AUDIT_v2.md (N2)
measured exactly that: with the original join on ``(season, round, driver, circuit)``
reinstated, ``avg_finish_at_remaining_circuits`` came back 100% null on the synthetic
fixture and ``test_features_are_unchanged_by_deleting_the_future`` still passed, because a
column that is null everywhere is trivially unchanged by deleting the future. Only a test
that pins the values can tell a correct block from a dead one.

The reference reads each driver's previous-season finishes at each circuit still on the
calendar, straight off the panel, with no joins to get wrong.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest

from f1cp.config.settings import FeaturesConfig, PathsConfig
from f1cp.features.pipeline import run_pipeline
from tests.unit.features.synthetic import SEASONS, build_synthetic_sources

pytestmark = pytest.mark.unit


@pytest.fixture
def panel(tmp_path: Path) -> pl.DataFrame:
    paths = PathsConfig(
        data=tmp_path,
        raw=tmp_path / "raw",
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        cache=tmp_path / "cache",
        artifacts=tmp_path / "artifacts",
        reports=tmp_path / "reports",
    )
    return run_pipeline(build_synthetic_sources(), FeaturesConfig(), paths)


def _reference(panel: pl.DataFrame, *, min_history: int, lookback: int) -> pl.DataFrame:
    """Average prior finish at the remaining circuits, one row at a time."""
    finishes: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in panel.iter_rows(named=True):
        if row["finish_position"] is not None:
            finishes[(row["driver_id"], row["circuit_id"])].append(
                (row["season"], row["finish_position"])
            )

    calendar: dict[int, dict[int, str]] = defaultdict(dict)
    for row in panel.select("season", "round", "circuit_id").unique().iter_rows(named=True):
        calendar[row["season"]][row["round"]] = row["circuit_id"]

    rows = []
    for row in panel.select("season", "round", "driver_id").iter_rows(named=True):
        season, round_number, driver = row["season"], row["round"], row["driver_id"]
        remaining = {c for r, c in calendar[season].items() if r > round_number}
        averages = []
        for circuit in remaining:
            prior = [
                position
                for visit_season, position in finishes[(driver, circuit)]
                if season - lookback <= visit_season < season
            ]
            if len(prior) >= min_history:
                averages.append(sum(prior) / len(prior))
        rows.append(
            {
                "season": season,
                "round": round_number,
                "driver_id": driver,
                "expected_avg": sum(averages) / len(averages) if averages else None,
                "expected_visited": len(averages),
            }
        )
    return pl.DataFrame(rows)


def test_matches_the_row_at_a_time_reference(panel: pl.DataFrame) -> None:
    config = FeaturesConfig().blocks.circuit
    # The fixture spans three seasons, so a lookback of 10 never truncates history. That
    # keeps this test about the join, not about how the window is counted.
    assert config.lookback_seasons >= len(SEASONS)
    expected = _reference(
        panel, min_history=config.min_history_races, lookback=config.lookback_seasons
    )
    joined = panel.join(expected, on=["season", "round", "driver_id"], how="inner")
    assert joined.height == panel.height

    assert (joined["remaining_circuits_visited"] == joined["expected_visited"]).all()
    assert (
        joined["avg_finish_at_remaining_circuits"].is_null() == joined["expected_avg"].is_null()
    ).all()
    both = joined.drop_nulls(["avg_finish_at_remaining_circuits", "expected_avg"])
    assert (both["avg_finish_at_remaining_circuits"] - both["expected_avg"]).abs().max() < 1e-9


def test_is_populated_once_drivers_have_history(panel: pl.DataFrame) -> None:
    """The failure mode N2 measured: a dead block is null everywhere, not just at first."""
    by_season = panel.group_by("season").agg(
        pl.col("avg_finish_at_remaining_circuits").is_not_null().mean().alias("coverage")
    )
    coverage = dict(by_season.iter_rows())
    assert coverage[SEASONS[0]] == 0.0, "no previous season to read in the first one"
    assert coverage[SEASONS[1]] > 0.5
    assert coverage[SEASONS[2]] > 0.5


def test_a_known_row(panel: pl.DataFrame) -> None:
    """Pinned by hand so a reference and block that drift together still fail.

    Before round 1 of 2021, all four circuits are still to come. d1's 2019-2020 finishes:
    circuit_a [3, 3, 2, 2] -> 2.5, circuit_b [2, 1, 1, 3] -> 1.75, circuit_c [1, 1] -> 1.0,
    circuit_d [2, 1] -> 1.5. The mean over the four is 6.75 / 4 = 1.6875.
    """
    row = panel.filter(
        (pl.col("season") == 2021) & (pl.col("round") == 1) & (pl.col("driver_id") == "d1")
    )
    assert row["remaining_circuits_visited"].item() == 4
    assert row["avg_finish_at_remaining_circuits"].item() == pytest.approx(1.6875)
