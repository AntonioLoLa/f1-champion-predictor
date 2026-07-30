"""Rolling form features, and specifically the closed-form ``trend`` slope.

``trend`` is the one form feature not expressible as a stock polars rolling aggregation,
so it carries its own arithmetic (see :func:`f1cp.features.form._rolling_ols_slope`) and
therefore its own tests. The reference implementation below is the obvious, slow,
row-at-a-time fit the closed form replaces; the equivalence test is what licenses that
replacement.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from f1cp.features.form import _rolling_ols_slope

pytestmark = pytest.mark.unit

GROUP = ["season", "driver_id"]


def _reference_slope(values: pl.Series) -> float | None:
    """Least-squares slope over a window, nulls dropped, fitted row-at-a-time."""
    positions = values.drop_nulls().to_numpy()
    if positions.shape[0] < 2:
        return None
    x = np.arange(positions.shape[0], dtype=float)
    slope, _ = np.polyfit(x, positions, 1)
    return float(slope)


def _reference_trend(frame: pl.DataFrame, *, window: int) -> pl.Series:
    """Apply :func:`_reference_slope` the naive way, over each driver's own history.

    The cast to ``Float64`` is load-bearing, and was the bug this reference exists to pin
    down: ``rolling_map`` returns a series of the *input* column's dtype, so fitting over
    an integer ``finish_position`` silently truncated every slope back to a whole number
    -- and since ``np.polyfit`` returns 2.999999999999999 where the exact answer is 3,
    that truncation was an off-by-one toward zero on most non-trivial windows.
    """
    return frame.select(
        pl.col("finish_position")
        .cast(pl.Float64)
        .shift(1)
        .rolling_map(_reference_slope, window_size=window, min_samples=2)
        .over(GROUP)
        .alias("trend")
    )["trend"]


def _frame(positions: list[int | None], *, season: int = 2020) -> pl.DataFrame:
    """One driver's season, given their finishing position round by round."""
    return pl.DataFrame(
        {
            "season": [season] * len(positions),
            "driver_id": ["driver"] * len(positions),
            "round": list(range(1, len(positions) + 1)),
            "finish_position": positions,
        },
        schema_overrides={"finish_position": pl.Int64},
    )


def _trend(frame: pl.DataFrame, *, window: int) -> list[float | None]:
    return frame.select(_rolling_ols_slope("finish_position", window=window, group=GROUP))[
        :, 0
    ].to_list()


def test_trend_is_the_slope_of_the_previous_finishes() -> None:
    # Rounds 1-4 finish 1st, 2nd, 3rd, 4th: coming into round 4 the previous three
    # results are 1, 2, 3 -- a slope of exactly +1, i.e. steadily losing places.
    trend = _trend(_frame([1, 2, 3, 4]), window=3)

    assert trend[0] is None  # nothing precedes the first round
    assert trend[1] is None  # a single previous result determines no slope
    assert trend[2] == pytest.approx(1.0)
    assert trend[3] == pytest.approx(1.0)


def test_trend_is_negative_when_finishes_are_improving() -> None:
    assert _trend(_frame([10, 6, 2]), window=3)[2] == pytest.approx(-4.0)


def test_trend_is_a_float_not_a_truncated_integer() -> None:
    """Finishes of 1, 2, 4 slope at +1.5, which an integer result would flatten to 1."""
    frame = _frame([1, 2, 4, 9])
    dtype = frame.select(_rolling_ols_slope("finish_position", window=3, group=GROUP)).dtypes[0]

    assert dtype == pl.Float64
    assert _trend(frame, window=3)[3] == pytest.approx(1.5)


def test_trend_fits_over_appearances_not_the_calendar() -> None:
    """Rounds the driver missed are dropped, not treated as a gap in the x axis.

    Positions 2 and 6 two rounds apart fit the same slope as positions 2 and 6 in
    consecutive rounds: the regressor is "their last few outings", not "rounds elapsed".
    """
    with_gap = _trend(_frame([2, None, 6, 1]), window=3)[3]
    without_gap = _trend(_frame([2, 6, 1]), window=2)[2]

    assert with_gap == pytest.approx(4.0)
    assert with_gap == pytest.approx(without_gap)


def test_trend_is_null_until_two_appearances_exist() -> None:
    # One appearance in the window, however many rounds it spans, fixes no slope.
    assert _trend(_frame([None, 5, None, None]), window=3) == [None, None, None, None]


def test_trend_does_not_run_across_drivers_or_seasons() -> None:
    frame = pl.concat(
        [
            _frame([1, 2, 3], season=2020),
            _frame([9, 8, 7], season=2021),
        ]
    )
    trend = _trend(frame, window=3)

    # The 2021 season restarts from scratch rather than continuing 2020's ascent.
    assert trend[3] is None
    assert trend[4] is None
    assert trend[5] == pytest.approx(-1.0)


@pytest.mark.parametrize("window", [2, 3, 5, 8])
def test_trend_matches_a_row_at_a_time_least_squares_fit(window: int) -> None:
    """The closed form must agree with the naive fit, nulls and partial windows included.

    Randomised over a season-shaped panel with a realistic share of missing rounds (a
    driver contests only some of them), since the null handling is where the two
    implementations could most easily diverge.
    """
    generator = np.random.default_rng(seed=17)
    n_rounds = 24
    drivers = [f"driver_{index}" for index in range(8)]

    rows = {"season": [], "driver_id": [], "round": [], "finish_position": []}
    for season in (2019, 2020):
        for driver in drivers:
            for round_number in range(1, n_rounds + 1):
                rows["season"].append(season)
                rows["driver_id"].append(driver)
                rows["round"].append(round_number)
                absent = generator.random() < 0.4
                rows["finish_position"].append(None if absent else int(generator.integers(1, 21)))

    frame = pl.DataFrame(rows, schema_overrides={"finish_position": pl.Int64}).sort(
        ["season", "driver_id", "round"]
    )

    actual = frame.select(
        _rolling_ols_slope("finish_position", window=window, group=GROUP).alias("trend")
    )["trend"]
    expected = _reference_trend(frame, window=window)

    assert (actual.is_null() == expected.is_null()).all()
    both = actual.is_not_null()
    assert both.sum() > 0
    np.testing.assert_allclose(
        actual.filter(both).to_numpy(), expected.filter(both).to_numpy(), rtol=1e-9, atol=1e-9
    )
