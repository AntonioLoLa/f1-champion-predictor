"""Driver-level features: age, experience, career record and defending-champion status.

These are the features a scout would reach for first: how old is this driver, how many
years have they been doing this, and have they done it before. Cheap to compute, and
surprisingly predictive in combination with the season-specific blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["DriverBlock"]


class DriverBlock(FeatureBlock):
    """Age, experience, career win rate and prior-championship features."""

    name = "driver"
    requires = ("season", "driver_id", "round", "race_date", "is_win", "entered")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Compute career-level features spanning every season in the panel.

        Args:
            frame: The panel built so far.
            context: Shared inputs; uses ``sources.drivers`` for birth dates.

        Returns:
            ``frame`` with driver-level columns appended.
        """
        drivers = context.sources.drivers.select(
            "driver_id", pl.col("date_of_birth").alias("_date_of_birth")
        )
        out = frame.join(drivers, on="driver_id", how="left").with_columns(
            ((pl.col("race_date") - pl.col("_date_of_birth")).dt.total_days() / 365.25).alias("age")
        )

        # Career figures span the driver's whole history, not just the current season, so
        # they are computed across the full panel (sorted season, driver, round) rather
        # than partitioned by season the way the in-season blocks are.
        career_group = ["driver_id"]
        out = out.sort(["driver_id", "season", "round"]).with_columns(
            pl.col("season").n_unique().over(career_group).alias("_career_seasons_total"),
            pl.col("entered")
            .cast(pl.Int32)
            .shift(1)
            .cum_sum()
            .over(career_group)
            .fill_null(0)
            .alias("career_starts"),
            pl.col("is_win")
            .cast(pl.Int32)
            .shift(1)
            .cum_sum()
            .over(career_group)
            .fill_null(0)
            .alias("career_wins"),
            pl.col("season").cum_count().over(career_group).alias("_row_index"),
        )

        first_round_of_season = out.group_by(["driver_id", "season"]).agg(
            pl.col("round").min().alias("_season_first_round")
        )
        out = out.join(first_round_of_season, on=["driver_id", "season"], how="left")
        seasons_so_far = (
            out.filter(pl.col("round") == pl.col("_season_first_round"))
            .sort(["driver_id", "season"])
            .with_columns(
                pl.col("season").cum_count().over("driver_id").alias("experience_seasons")
            )
            .select("driver_id", "season", "experience_seasons")
        )
        out = out.join(seasons_so_far, on=["driver_id", "season"], how="left").with_columns(
            pl.col("experience_seasons").forward_fill().over("driver_id"),
        )

        out = out.with_columns(
            pl.when(pl.col("career_starts") > 0)
            .then(pl.col("career_wins") / pl.col("career_starts"))
            .otherwise(0.0)
            .alias("career_win_rate"),
        )

        champions = context.sources.driver_standings.filter(pl.col("position") == 1)
        final_rounds = context.sources.races.group_by("season").agg(
            pl.col("round").max().alias("_final_round")
        )
        title_seasons = (
            champions.join(
                context.sources.races.select("season", "round"),
                on=["season", "round"],
                how="inner",
            )
            .join(final_rounds, on="season", how="inner")
            .filter(pl.col("round") == pl.col("_final_round"))
            .select("driver_id", pl.col("season").alias("_title_season"))
        )

        # `title_seasons` has one row per (driver, title season), so a multi-time champion
        # appears more than once. Aggregating a *separate* small frame down to one row per
        # (driver, season) before joining -- rather than joining title_seasons directly
        # onto the full panel -- avoids fanning out every panel row for that driver.
        seasons_in_panel = out.select("driver_id", "season").unique()
        prior_titles = (
            seasons_in_panel.join(title_seasons, on="driver_id", how="inner")
            .filter(pl.col("_title_season") < pl.col("season"))
            .group_by(["driver_id", "season"])
            .agg(pl.col("_title_season").n_unique().alias("prior_titles"))
        )
        out = out.join(prior_titles, on=["driver_id", "season"], how="left").with_columns(
            pl.col("prior_titles").fill_null(0)
        )

        defending = (
            title_seasons.with_columns((pl.col("_title_season") + 1).alias("season"))
            .select("driver_id", "season", pl.lit(value=True).alias("is_defending_champion"))
            .unique()
        )
        out = out.join(defending, on=["driver_id", "season"], how="left").with_columns(
            pl.col("is_defending_champion").fill_null(value=False)
        )

        return out.drop(
            "_date_of_birth", "_career_seasons_total", "_row_index", "_season_first_round"
        ).sort(["season", "driver_id", "round"])
