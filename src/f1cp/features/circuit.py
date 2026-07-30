"""Circuit historical-performance features.

A driver's average finish at the circuits still left on the calendar is a small, targeted
piece of foreknowledge: a driver strong at the remaining tracks has an edge that generic
form features cannot see, because it depends on *which* races are left to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock
from f1cp.features.expr_utils import cum_mean

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

        # A driver's per-circuit history, as an expanding average *before* the round it is
        # attached to -- built once, independent of the calling row's own season/round, so
        # it can be looked up for any future circuit on the calendar.
        history = (
            frame.select("season", "round", "driver_id", "circuit_id", "finish_position")
            .sort(["driver_id", "circuit_id", "season", "round"])
            .with_columns(
                cum_mean(pl.col("finish_position").shift(1))
                .over(["driver_id", "circuit_id"])
                .alias("_prior_avg_at_circuit"),
                pl.col("finish_position")
                .shift(1)
                .is_not_null()
                .cast(pl.Int32)
                .cum_sum()
                .over(["driver_id", "circuit_id"])
                .alias("_prior_visits_at_circuit"),
            )
            .select(
                "season",
                "round",
                "driver_id",
                "circuit_id",
                "_prior_avg_at_circuit",
                "_prior_visits_at_circuit",
            )
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

        enriched = drivers_x_remaining.join(
            history, on=["season", "round", "driver_id", "circuit_id"], how="left"
        ).filter(pl.col("_prior_visits_at_circuit") >= block_config.min_history_races)

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
