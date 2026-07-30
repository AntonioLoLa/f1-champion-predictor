"""Championship standings features.

Points, position and gap-to-leader read straight from the provider's standings table
rather than recomputed from race results, because seasons up to 1990 used "best N results"
scoring: a driver's championship total is not the sum of their race points. See
:mod:`f1cp.domain.scoring` for the full rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["StandingsBlock"]


class StandingsBlock(FeatureBlock):
    """Points, position and gaps to the leader, with era-normalised twins."""

    name = "standings"
    requires = ("season", "round", "driver_id", "round_max_points")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Join provider standings onto the panel and derive gap features.

        Args:
            frame: The panel built so far.
            context: Shared inputs; uses ``sources.driver_standings``.

        Returns:
            ``frame`` with standings columns appended.
        """
        standings = context.sources.driver_standings
        block_config = context.config.blocks.standings

        joined = frame.join(
            standings.select(
                "season",
                "round",
                "driver_id",
                pl.col("points").alias("current_points"),
                pl.col("position").alias("championship_position"),
                pl.col("wins").alias("season_wins"),
            ),
            on=["season", "round", "driver_id"],
            how="left",
        ).with_columns(
            pl.col("current_points").fill_null(0.0),
            pl.col("season_wins").fill_null(0),
        )

        leader_points = (
            joined.filter(pl.col("championship_position") == 1).select(
                "season", "round", pl.col("current_points").alias("_leader_points")
            )
            # Shared drives in the 1950s produced the occasional tie at P1; keep one row.
            .unique(subset=["season", "round"])
        )
        second_points = (
            joined.filter(pl.col("championship_position") == 2)
            .select("season", "round", pl.col("current_points").alias("_second_points"))
            .unique(subset=["season", "round"])
        )
        points_awarded = joined.group_by(["season", "round"]).agg(
            pl.col("current_points").sum().alias("_points_awarded_to_date")
        )

        joined = (
            joined.join(leader_points, on=["season", "round"], how="left")
            .join(second_points, on=["season", "round"], how="left")
            .join(points_awarded, on=["season", "round"], how="left")
            .with_columns(
                (pl.col("_leader_points") - pl.col("current_points")).alias("gap_to_leader"),
                (pl.col("_second_points") - pl.col("current_points")).alias("gap_to_second"),
                (pl.col("championship_position") == 1).fill_null(value=False).alias("is_leader"),
                pl.when(pl.col("_points_awarded_to_date") > 0)
                .then(pl.col("current_points") / pl.col("_points_awarded_to_date"))
                .otherwise(0.0)
                .alias("points_share"),
            )
            .drop("_leader_points", "_second_points", "_points_awarded_to_date")
        )

        if block_config.era_normalisation:
            joined = (
                joined.join(_points_for_win_by_season(joined), on="season", how="left")
                .with_columns(
                    (pl.col("current_points") / pl.col("_points_for_win")).alias(
                        "current_points_norm"
                    ),
                    (pl.col("gap_to_leader") / pl.col("_points_for_win")).alias(
                        "gap_to_leader_norm"
                    ),
                    (pl.col("gap_to_second") / pl.col("_points_for_win")).alias(
                        "gap_to_second_norm"
                    ),
                )
                .drop("_points_for_win")
            )

        return joined


def _points_for_win_by_season(frame: pl.DataFrame) -> pl.DataFrame:
    from f1cp.features.panel import build_scoring_reference

    seasons = frame["season"].unique().to_list()
    return build_scoring_reference(seasons).select(
        "season", pl.col("points_for_win").alias("_points_for_win")
    )
