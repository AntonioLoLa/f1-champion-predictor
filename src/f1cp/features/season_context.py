"""Season progress and remaining-points features.

``gap_to_leader / max_points_remaining`` is the single most informative ratio the dataset
carries: it directly answers "can this driver still mathematically win", which is exactly
what ``is_mathematically_eliminated`` makes explicit as a boolean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["SeasonContextBlock"]


class SeasonContextBlock(FeatureBlock):
    """Round number, season progress and mathematical-elimination features."""

    name = "season_context"
    requires = ("season", "round", "rounds_total", "max_points_remaining", "gap_to_leader")

    def build(self, frame: pl.DataFrame, _context: FeatureContext) -> pl.DataFrame:
        """Derive progress and elimination features from calendar and standings columns.

        Args:
            frame: The panel built so far. Must already carry ``gap_to_leader`` (from
                :class:`~f1cp.features.standings.StandingsBlock`) and the calendar columns
                attached by :func:`f1cp.features.panel.build_panel`.
            _context: Shared inputs; unused, this block only reads panel columns.

        Returns:
            ``frame`` with season-progress columns appended.
        """
        return frame.with_columns(
            pl.col("rounds_total").alias("rounds_total"),
            (pl.col("rounds_total") - pl.col("round")).alias("rounds_remaining"),
            (pl.col("round") / pl.col("rounds_total")).alias("season_progress"),
            (pl.col("gap_to_leader") > pl.col("max_points_remaining"))
            .fill_null(value=False)
            .alias("is_mathematically_eliminated"),
        ).with_columns(
            pl.when(pl.col("max_points_remaining") > 0)
            .then(pl.col("gap_to_leader") / pl.col("max_points_remaining"))
            .when(pl.col("gap_to_leader") <= 0)
            .then(0.0)
            .otherwise(1.0)
            .alias("gap_to_leader_over_remaining"),
        )
