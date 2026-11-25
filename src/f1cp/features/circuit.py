"""Circuit historical-performance features.

A driver's average finish at the circuits still left on the calendar is a small, targeted
piece of foreknowledge: a driver strong at the remaining tracks has an edge that generic
form features cannot see, because it depends on *which* races are left to run.

The lookup is over *previous seasons only*. A circuit that is still on the calendar has, by
definition, not been raced yet this year, so there is nothing from the current season to
read -- and reading it would be a lookahead. That leaves the feature genuinely sparse in a
driver's first seasons, which is honest: a rookie has no record at Suzuka.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["CircuitBlock"]


class CircuitBlock(FeatureBlock):
    """Historical average finish at the circuits remaining on the calendar."""

    name = "circuit"
    requires = ("season", "round", "driver_id", "circuit_id", "rounds_total")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Compute remaining-circuit historical form.

        Args:
            frame: The panel built so far.
            context: Shared inputs; reads ``config.blocks.circuit`` and the full race
                calendar (including future rounds) to know which circuits remain.

        Returns:
            ``frame`` with ``avg_finish_at_remaining_circuits``,
            ``remaining_circuits_visited`` and ``remaining_home_races`` appended.
        """
        block_config = context.config.blocks.circuit
        races = context.sources.races.select("season", "round", "circuit_id")
        drivers = (
            frame.select("driver_id")
            .unique()
            .join(context.sources.drivers, on="driver_id", how="left")
            .select("driver_id", pl.col("nationality").alias("_driver_nationality"))
        )
        circuits = context.sources.circuits.select(
            "circuit_id", pl.col("country").alias("_circuit_country")
        )

        # A driver's record at each circuit, as of the end of each season they visited it.
        # Keyed by the season the history was accumulated *through*, not by the round the
        # circuit happened to fall on -- the lookup below asks "what did this driver know
        # about Monza before this season", and Monza's round number is irrelevant to that
        # question. Joining on the round as well would only match when a circuit is raced
        # twice in one season, leaving the block almost entirely null.
        visits = (
            frame.filter(pl.col("finish_position").is_not_null())
            .group_by(["driver_id", "circuit_id", "season"])
            .agg(
                pl.col("finish_position").sum().alias("_season_positions"),
                pl.len().alias("_season_visits"),
            )
            .sort(["driver_id", "circuit_id", "season"])
        )
        history = visits.with_columns(
            pl.col("_season_positions")
            .cum_sum()
            .over(["driver_id", "circuit_id"])
            .alias("_positions_through_season"),
            pl.col("_season_visits")
            .cum_sum()
            .over(["driver_id", "circuit_id"])
            .alias("_visits_through_season"),
        ).select(
            "driver_id",
            "circuit_id",
            pl.col("season").alias("_history_season"),
            "_positions_through_season",
            "_visits_through_season",
        )

        remaining = (
            races.join(
                frame.select("season", "round").unique().rename({"round": "_as_of_round"}),
                on="season",
                how="inner",
            )
            .filter(pl.col("round") > pl.col("_as_of_round"))
            .select("season", pl.col("_as_of_round").alias("round"), "circuit_id")
        )

        drivers_x_remaining = (
            frame.select("season", "round", "driver_id")
            .unique()
            .join(remaining, on=["season", "round"], how="inner")
        )

        # Strictly previous seasons only. Within the current season the driver may not have
        # reached that circuit yet -- it is on the *remaining* calendar, after all -- and
        # reading a partial current-season aggregate here would be a lookahead.
        lookback = block_config.lookback_seasons
        enriched = (
            drivers_x_remaining.join(history, on=["driver_id", "circuit_id"], how="inner")
            .filter(
                (pl.col("_history_season") < pl.col("season"))
                & (pl.col("_history_season") >= pl.col("season") - lookback)
            )
            .group_by(["season", "round", "driver_id", "circuit_id"])
            .agg(
                pl.col("_positions_through_season")
                .sort_by("_history_season")
                .last()
                .alias("_positions"),
                pl.col("_visits_through_season").sort_by("_history_season").last().alias("_visits"),
            )
            .filter(pl.col("_visits") >= block_config.min_history_races)
            .with_columns((pl.col("_positions") / pl.col("_visits")).alias("_prior_avg_at_circuit"))
        )

        summary = enriched.group_by(["season", "round", "driver_id"]).agg(
            pl.col("_prior_avg_at_circuit").mean().alias("avg_finish_at_remaining_circuits"),
            pl.col("circuit_id").n_unique().alias("remaining_circuits_visited"),
        )

        home = (
            remaining.join(circuits, on="circuit_id", how="left")
            .join(drivers, how="cross")
            .filter(pl.col("_circuit_country") == pl.col("_driver_nationality"))
            .group_by(["season", "round", "driver_id"])
            .agg(pl.col("circuit_id").n_unique().alias("remaining_home_races"))
        )

        return (
            frame.join(summary, on=["season", "round", "driver_id"], how="left")
            .join(home, on=["season", "round", "driver_id"], how="left")
            .with_columns(
                pl.col("remaining_circuits_visited").fill_null(0),
                pl.col("remaining_home_races").fill_null(0),
            )
        )
