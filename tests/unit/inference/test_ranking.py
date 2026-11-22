"""The served ranking within a round, and how it breaks ties."""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.inference.predictor import rank_within_rounds

pytestmark = pytest.mark.unit

_GROUPS = ("season", "round")


def _round(rows: list[tuple[str, float, int | None]]) -> pl.DataFrame:
    """One round of ``(driver_id, rescaled probability, championship position)`` rows."""
    return pl.DataFrame(
        {
            "season": [2026] * len(rows),
            "round": [20] * len(rows),
            "driver_id": [driver for driver, _, _ in rows],
            "driver_name": [driver.title() for driver, _, _ in rows],
            "probability_normalized": [probability for _, probability, _ in rows],
            "championship_position": [position for _, _, position in rows],
        },
        schema_overrides={"championship_position": pl.Int64},
    )


def _order(frame: pl.DataFrame) -> list[str]:
    return frame.sort("rank")["driver_id"].to_list()


def test_a_decided_title_ranks_the_rest_of_the_field_by_the_standings() -> None:
    # The 2026 Brazilian GP: everyone but the leader eliminated and zeroed. Alphabetical
    # row order used to put Albon second instead of Verstappen.
    frame = _round(
        [
            ("albon", 0.0, 15),
            ("antonelli", 1.0, 1),
            ("hamilton", 0.0, 4),
            ("russell", 0.0, 3),
            ("verstappen", 0.0, 2),
        ]
    )

    ranked = rank_within_rounds(frame, group_columns=_GROUPS)

    assert _order(ranked) == ["antonelli", "verstappen", "russell", "hamilton", "albon"]
    assert sorted(ranked["rank"].to_list()) == [1, 2, 3, 4, 5]


def test_the_models_order_survives_a_near_tie_against_the_standings() -> None:
    frame = _round([("leader", 0.0001, 1), ("chaser", 0.0002, 2)])

    ranked = rank_within_rounds(frame, group_columns=_GROUPS)

    assert _order(ranked) == ["chaser", "leader"]


def test_a_driver_without_a_standing_ranks_after_every_classified_tie() -> None:
    frame = _round([("aaa", 0.0, None), ("zzz", 0.0, 12), ("top", 1.0, 1)])

    assert _order(rank_within_rounds(frame, group_columns=_GROUPS)) == ["top", "zzz", "aaa"]


def test_ranks_restart_in_every_round() -> None:
    frame = pl.concat(
        [
            _round([("a", 0.6, 1), ("b", 0.4, 2)]),
            _round([("a", 0.3, 2), ("b", 0.7, 1)]).with_columns(
                pl.lit(21, dtype=pl.Int64).alias("round")
            ),
        ]
    )

    ranked = rank_within_rounds(frame, group_columns=_GROUPS)

    by_round = {
        round_number: _order(ranked.filter(pl.col("round") == round_number))
        for round_number in (20, 21)
    }
    assert by_round == {20: ["a", "b"], 21: ["b", "a"]}
    assert ranked.filter(pl.col("round") == 21)["rank"].sort().to_list() == [1, 2]
