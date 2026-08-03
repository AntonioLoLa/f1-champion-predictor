"""Teammate comparison, and specifically that it compares finishing position.

Comparing round points made two team-mates who both scored zero -- common outside the
points-paying positions, and also true of any DNF -- read as "beat neither", discarding a
real gap like P11 vs P16, or a classified finish vs a retirement.

``teammate_beat_rate`` is the *shifted* cumulative mean of the per-round comparison, so a
round's outcome is only observable at the following round -- these tests build two rounds
and read the second one.
"""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.features.constructor import _teammate_features

pytestmark = pytest.mark.unit


def _two_round_frame(round_one_positions: tuple[int | None, int | None]) -> pl.DataFrame:
    """One team-mate pair, `d1` and `d2`, with a given round-1 finishing order."""
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020],
            "round": [1, 1, 2, 2],
            "constructor_id": ["alpha", "alpha", "alpha", "alpha"],
            "driver_id": ["d1", "d2", "d1", "d2"],
            "current_points": [0.0, 0.0, 0.0, 0.0],
            "finish_position": [*round_one_positions, 5, 6],
        },
        schema_overrides={"finish_position": pl.Int64},
    )


def _rates_at_round_two(frame: pl.DataFrame) -> dict[str, float]:
    round_two = _teammate_features(frame).filter(pl.col("round") == 2)
    return dict(
        zip(
            round_two["driver_id"].to_list(), round_two["teammate_beat_rate"].to_list(), strict=True
        )
    )


def test_a_tied_points_round_is_still_decided_by_finishing_position() -> None:
    """Both score zero (P11, P16), but d1 finished ahead and must be credited the win."""
    rates = _rates_at_round_two(_two_round_frame((11, 16)))

    assert rates["d1"] == pytest.approx(1.0)
    assert rates["d2"] == pytest.approx(0.0)


def test_a_dnf_never_beats_a_classified_teammate_even_if_both_score_zero() -> None:
    """d1 retires (null position), d2 finishes P15 -- classified beats retired, not a tie."""
    rates = _rates_at_round_two(_two_round_frame((None, 15)))

    assert rates["d1"] == pytest.approx(0.0)
    assert rates["d2"] == pytest.approx(1.0)


def test_two_dnfs_in_the_same_round_are_a_tie() -> None:
    """Neither retirement counts as beating the other."""
    rates = _rates_at_round_two(_two_round_frame((None, None)))

    assert rates["d1"] == pytest.approx(0.0)
    assert rates["d2"] == pytest.approx(0.0)
