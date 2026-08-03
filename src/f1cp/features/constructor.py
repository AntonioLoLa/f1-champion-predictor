"""Constructor strength and teammate comparison features.

Two drivers in the same car are the cleanest natural experiment Formula 1 offers: nearly
everything but driver skill is held constant, which makes ``teammate_points_delta`` and
``teammate_beat_rate`` some of the strongest signals available for title contention, well
beyond what raw championship position captures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock
from f1cp.features.expr_utils import cum_mean

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["ConstructorBlock"]


class ConstructorBlock(FeatureBlock):
    """Team standings, reliability, form and teammate head-to-head."""

    name = "constructor"
    requires = (
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "current_points",
        "round_points",
        "finish_position",
        "is_dnf",
        "entered",
    )

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Join constructor standings and derive teammate comparison features.

        Args:
            frame: The panel built so far. Must carry ``current_points`` (driver
                standings, from :class:`~f1cp.features.standings.StandingsBlock`).
            context: Shared inputs; uses ``sources.constructor_standings``.

        Returns:
            ``frame`` with constructor-level and teammate columns appended.
        """
        constructor_standings = context.sources.constructor_standings
        block_config = context.config.blocks.constructor

        out = frame.join(
            constructor_standings.select(
                "season",
                "round",
                "constructor_id",
                pl.col("points").alias("team_points"),
                pl.col("position").alias("team_position"),
                pl.col("wins").alias("team_wins"),
            ),
            on=["season", "round", "constructor_id"],
            how="left",
        ).with_columns(
            pl.col("team_points").fill_null(0.0),
        )

        team_points_awarded = out.group_by(["season", "round"]).agg(
            pl.col("team_points")
            .filter(pl.col("team_position").is_not_null())
            .unique()
            .sum()
            .alias("_team_points_awarded")
        )
        out = (
            out.join(team_points_awarded, on=["season", "round"], how="left")
            .with_columns(
                pl.when(pl.col("_team_points_awarded") > 0)
                .then(pl.col("team_points") / pl.col("_team_points_awarded"))
                .otherwise(0.0)
                .alias("team_points_share"),
            )
            .drop("_team_points_awarded")
        )

        # No `team_reliability` twin of `team_dnf_rate`: it was exactly `1 - team_dnf_rate`,
        # and a pair of perfectly anti-correlated columns halves the importance each is
        # credited with while adding nothing a model can use.
        out = out.join(
            _team_round_rolling(out), on=["season", "round", "constructor_id"], how="left"
        )

        if block_config.teammate_head_to_head:
            out = _teammate_features(out)

        return out


def _team_round_rolling(frame: pl.DataFrame) -> pl.DataFrame:
    """Team-level rolling reliability and recent-points form, one row per team round.

    A team's two cars each get a row per round in ``frame``, which -- sorted by
    ``(driver_id, round)`` -- is not chronological within a constructor. Rolling and
    cumulative windows are therefore computed here on a frame collapsed to one row per
    ``(season, round, constructor_id)`` and explicitly sorted by round first, then joined
    back onto every driver row of that team.
    """
    team_group = ["season", "constructor_id"]
    per_round = (
        frame.group_by(["season", "round", "constructor_id"])
        .agg(
            pl.col("round_points").sum().alias("_team_round_points"),
            pl.col("is_dnf").sum().alias("_team_round_dnfs"),
            pl.col("entered").sum().alias("_team_round_entries"),
        )
        .sort([*team_group, "round"])
    )
    return per_round.with_columns(
        pl.col("_team_round_points")
        .shift(1)
        .rolling_sum(window_size=3, min_samples=1)
        .over(team_group)
        .alias("team_points_last_3"),
        (
            pl.col("_team_round_dnfs").cast(pl.Int32).shift(1).cum_sum().over(team_group)
            / pl.col("_team_round_entries")
            .cast(pl.Int32)
            .shift(1)
            .cum_sum()
            .over(team_group)
            .clip(lower_bound=1)
        )
        .fill_null(0.0)
        .alias("team_dnf_rate"),
    ).select("season", "round", "constructor_id", "team_points_last_3", "team_dnf_rate")


#: Sentinel finishing position for a DNF, worse than any position a classified car can
#: hold -- so a retirement never counts as "beating" a teammate who finished, and two
#: retirements in the same round are a tie rather than a spurious win for either side.
_DNF_SENTINEL_POSITION = 999


def _teammate_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Compare each driver against their team-mate(s) in the same round.

    A driver line-up can change mid-season, so "team-mate" is resolved per round rather
    than per season: whoever else drove for the same constructor in that specific round.

    Compared on finishing position, not round points: two team-mates outside the points
    (common for the median team on the grid) both score zero, which made every such round
    read as "beat neither", discarding the real gap between e.g. P11 and P16.
    """
    roster = frame.select(
        "season", "round", "constructor_id", "driver_id", "current_points", "finish_position"
    )
    paired = roster.join(
        roster.rename(
            {
                "driver_id": "_teammate_id",
                "current_points": "_teammate_points",
                "finish_position": "_teammate_finish_position",
            }
        ),
        on=["season", "round", "constructor_id"],
        how="left",
    ).filter(pl.col("driver_id") != pl.col("_teammate_id"))

    per_round = paired.group_by(["season", "round", "driver_id"]).agg(
        pl.col("_teammate_points").mean().alias("_teammate_points_avg"),
        (
            pl.col("finish_position").fill_null(_DNF_SENTINEL_POSITION)
            < pl.col("_teammate_finish_position").fill_null(_DNF_SENTINEL_POSITION)
        )
        .mean()
        .alias("_beat_teammate_this_round"),
    )

    group = ["season", "driver_id"]
    out = (
        frame.join(per_round, on=["season", "round", "driver_id"], how="left")
        .with_columns(
            (pl.col("current_points") - pl.col("_teammate_points_avg")).alias(
                "teammate_points_delta"
            ),
            cum_mean(pl.col("_beat_teammate_this_round").shift(1))
            .over(group)
            .fill_null(0.5)
            .alias("teammate_beat_rate"),
        )
        .drop("_teammate_points_avg", "_beat_teammate_this_round")
    )

    return out.with_columns(pl.col("teammate_points_delta").fill_null(0.0))
