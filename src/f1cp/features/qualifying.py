"""Qualifying and grid-position features.

``grid`` (the position a driver actually started from, after any penalties) and
``quali_position`` (the raw qualifying classification) are kept as separate signals: a
driver who qualified on pole but starts P10 after a gearbox penalty is a different story to
one who simply qualified P10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock
from f1cp.features.expr_utils import cum_mean

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["QualifyingBlock"]


class QualifyingBlock(FeatureBlock):
    """Season and recent-window averages of grid and qualifying position."""

    name = "qualifying"
    requires = ("season", "driver_id", "round", "grid", "quali_position", "positions_gained")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Compute expanding and recent-window qualifying features.

        Args:
            frame: The panel built so far.
            context: Shared inputs; reads ``config.blocks.qualifying.recent_window``.

        Returns:
            ``frame`` with qualifying columns appended.
        """
        group = ["season", "driver_id"]
        window = context.config.blocks.qualifying.recent_window

        return frame.with_columns(
            cum_mean(pl.col("grid").shift(1)).over(group).alias("season_avg_grid"),
            cum_mean(pl.col("quali_position").shift(1)).over(group).alias("season_avg_quali"),
            pl.col("grid")
            .shift(1)
            .rolling_mean(window_size=window, min_samples=1)
            .over(group)
            .alias(f"last_{window}_avg_grid"),
            cum_mean(pl.col("positions_gained").shift(1))
            .over(group)
            .alias("season_avg_positions_gained"),
            pl.col("grid").shift(1).cum_min().over(group).alias("season_best_grid"),
        )
