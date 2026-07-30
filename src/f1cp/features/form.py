"""Rolling form and momentum features.

Every window here is computed with ``shift(1)`` before the rolling aggregation, so a row
for round ``R`` summarises rounds strictly before the points scored *at* ``R`` -- i.e. it
answers "how was this driver doing coming into this race", not "how did they do including
this race". That shift is the difference between a legitimate predictor and a leaky one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["FormBlock"]


class FormBlock(FeatureBlock):
    """Rolling points/finish windows, momentum, trend and consistency."""

    name = "form"
    requires = ("season", "driver_id", "round", "round_points", "finish_position", "is_win")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Compute rolling-window form features per driver, within each season.

        Args:
            frame: The panel built so far, sorted by ``(season, driver_id, round)``.
            context: Shared inputs; reads ``config.blocks.form`` for window sizes.

        Returns:
            ``frame`` with form columns appended.
        """
        block_config = context.config.blocks.form
        group = ["season", "driver_id"]
        out = frame

        for window in block_config.windows:
            out = out.with_columns(
                pl.col("round_points")
                .shift(1)
                .rolling_sum(window_size=window, min_samples=1)
                .over(group)
                .alias(f"last_{window}_points"),
                pl.col("finish_position")
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over(group)
                .alias(f"last_{window}_avg_finish"),
            )

        out = out.with_columns(
            # Points-per-round coming into this race vs. the season average coming into
            # this race: positive momentum means the driver is outperforming their own
            # season so far, which a plain rolling average cannot distinguish from a
            # consistently strong driver.
            (
                pl.col("round_points").shift(1).cum_sum().over(group)
                / pl.col("round").shift(1).over(group).clip(lower_bound=1)
            ).alias("season_points_rate"),
        ).with_columns(
            (pl.col(f"last_{block_config.windows[0]}_points") / block_config.windows[0]).alias(
                "_recent_rate"
            ),
        )

        trend_window = block_config.trend_window
        out = out.with_columns(
            (pl.col("_recent_rate") - pl.col("season_points_rate")).alias("momentum"),
            _rolling_ols_slope("finish_position", window=trend_window, group=group).alias("trend"),
            pl.col("finish_position")
            .shift(1)
            .rolling_std(window_size=trend_window, min_samples=2)
            .over(group)
            .fill_null(0.0)
            .alias("consistency"),
        ).drop("_recent_rate")

        return out.with_columns(
            _rounds_since("is_win", group).alias("rounds_since_win"),
            _rounds_since("is_podium", group).alias("rounds_since_podium"),
        )


def _rolling_ols_slope(column: str, *, window: int, group: list[str]) -> pl.Expr:
    """Slope of an ordinary least-squares fit over the previous ``window`` values.

    Negative means finishing position is falling (improving), matching the "more negative
    is better" convention the other position features share.

    Nulls -- rounds a driver did not contest, the majority of panel rows -- are dropped
    before fitting, so the regressor is position within the driver's own last few
    appearances, not within the calendar.

    Closed-form column arithmetic rather than ``rolling_map``, which hands every one of the
    panel's tens of thousands of rows to a Python callback and cost ~4s of a ~5.5s
    prediction. Equivalence is verified row-for-row in ``tests/unit/features/test_form.py``.

    Args:
        column: Column to fit, e.g. ``finish_position``.
        window: How many previous rows to fit over.
        group: Columns to partition within (season, driver).

    Returns:
        An expression yielding the fitted slope, null where the window holds fewer than
        two non-null values and the fit is undetermined.
    """
    # Chronological order: the oldest lag first, so x = 0 is the earliest appearance.
    lags = [pl.col(column).shift(offset).over(group) for offset in range(window, 0, -1)]
    present = [lag.is_not_null() for lag in lags]

    # x is the position among the *non-null* values, i.e. how many appearances precede
    # this one inside the window -- which is what dropping nulls before fitting means.
    positions: list[pl.Expr] = []
    seen: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for flag in present:
        positions.append(seen)
        seen = seen + flag.cast(pl.Int64)

    # Masked to zero where absent, so a missing round contributes nothing to any sum.
    def _masked(values: list[pl.Expr]) -> list[pl.Expr]:
        return [
            pl.when(flag).then(value).otherwise(0.0).cast(pl.Float64)
            for flag, value in zip(present, values, strict=True)
        ]

    x = _masked(positions)
    y = _masked(lags)

    n = seen.cast(pl.Float64)
    sum_x = pl.sum_horizontal(x)
    sum_y = pl.sum_horizontal(y)
    sum_xy = pl.sum_horizontal([xi * yi for xi, yi in zip(x, y, strict=True)])
    sum_xx = pl.sum_horizontal([xi * xi for xi in x])

    return (
        pl.when(seen >= 2)
        .then((n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x))
        .otherwise(None)
    )


def _rounds_since(flag_column: str, group: list[str]) -> pl.Expr:
    """Rounds elapsed since ``flag_column`` was last true, as of the *previous* round.

    Args:
        flag_column: Boolean column marking the event (e.g. ``is_win``).
        group: Columns to partition the count within (season, driver).

    Returns:
        An expression counting rounds since the last true value, capped by the driver's
        rounds completed so far when the event has not yet occurred this season.
    """
    event_round = (
        pl.when(pl.col(flag_column))
        .then(pl.col("round"))
        .otherwise(None)
        .forward_fill()
        .over(group)
    )
    shifted_event_round = event_round.shift(1).over(group)
    shifted_round = pl.col("round").shift(1).over(group)
    return (
        pl.when(shifted_event_round.is_not_null())
        .then(shifted_round - shifted_event_round)
        .otherwise(shifted_round)
        .fill_null(0)
    )
