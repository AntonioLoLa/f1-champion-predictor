"""Construction of the dense driver-round panel.

Every feature block computes over one shared frame rather than re-joining the source
tables. Two properties of that frame do most of the work:

*Dense in rounds.* A driver who misses round 7 still gets a row for round 7, with the
round's contributions null. Without this, a "last 3 rounds" window would silently reach
back five rounds for a driver who skipped two, and the feature would mean something
different for different drivers.

*Sorted by driver then round.* Cumulative and rolling window expressions are therefore
well defined and, critically, only ever look backwards -- a row for round ``R`` aggregates
rounds ``1..R`` inclusive. Round ``R`` is the present, not the future: the dataset grain is
"the driver's state immediately after that Grand Prix".
"""

from __future__ import annotations

import polars as pl

from f1cp.domain.scoring import max_points_per_round, points_for_win, sprint_points_for_win
from f1cp.features.context import SourceFrames

__all__ = ["build_calendar", "build_panel", "build_scoring_reference"]

_PODIUM_POSITION = 3
_TOP_FIVE_POSITION = 5


def build_scoring_reference(seasons: list[int]) -> pl.DataFrame:
    """Tabulate the era's points system for each season.

    Points totals are not comparable across eras -- nine points for a win in 1985 and
    twenty-five in 2026 describe the same achievement -- so every points-derived feature
    gets a twin divided by these values.

    Args:
        seasons: Championship years to tabulate.

    Returns:
        A frame of ``season``, ``points_for_win`` and ``sprint_points_for_win``.
    """
    ordered = sorted(set(seasons))
    return pl.DataFrame(
        {
            "season": ordered,
            "points_for_win": [float(points_for_win(season)) for season in ordered],
            "sprint_points_for_win": [float(sprint_points_for_win(season)) for season in ordered],
        },
        schema={
            "season": pl.Int64,
            "points_for_win": pl.Float64,
            "sprint_points_for_win": pl.Float64,
        },
    )


def build_calendar(races: pl.DataFrame) -> pl.DataFrame:
    """Derive per-round season context from the full race calendar.

    The calendar includes rounds that have not yet been run, which is what makes
    ``max_points_remaining`` exact rather than an estimate: at round 11 of 2026 the
    remaining eleven rounds are known, and so is which of them carry a sprint race.

    Args:
        races: The ``races`` source frame, one row per scheduled Grand Prix.

    Returns:
        The calendar with ``rounds_total``, ``round_max_points`` and
        ``max_points_remaining`` added.
    """
    if not races.height:
        return races

    seasons = races["season"].unique().to_list()
    round_points = pl.DataFrame(
        {
            "season": [season for season in seasons for _ in (0, 1)],
            "has_sprint": [flag for _ in seasons for flag in (False, True)],
            "round_max_points": [
                float(max_points_per_round(season, is_sprint_weekend=flag))
                for season in seasons
                for flag in (False, True)
            ],
        },
        schema={"season": pl.Int64, "has_sprint": pl.Boolean, "round_max_points": pl.Float64},
    )

    calendar = (
        races.join(round_points, on=["season", "has_sprint"], how="left")
        .sort(["season", "round"])
        .with_columns(
            pl.col("round").max().over("season").alias("rounds_total"),
            pl.col("round_max_points").sum().over("season").alias("_season_max_points"),
            pl.col("round_max_points").cum_sum().over("season").alias("_max_points_to_date"),
        )
    )
    return calendar.with_columns(
        (pl.col("_season_max_points") - pl.col("_max_points_to_date")).alias("max_points_remaining")
    ).drop("_season_max_points", "_max_points_to_date")


def _build_entries(sources: SourceFrames) -> pl.DataFrame:
    """One row per driver per round they actually took part in."""
    sprint = (
        sources.sprint_results.select(
            "season",
            "round",
            "driver_id",
            pl.col("points").alias("sprint_points"),
            pl.col("position").alias("sprint_position"),
        )
        if sources.sprint_results.height
        else pl.DataFrame(
            schema={
                "season": pl.Int64,
                "round": pl.Int64,
                "driver_id": pl.String,
                "sprint_points": pl.Float64,
                "sprint_position": pl.Int64,
            }
        )
    )
    qualifying = (
        sources.qualifying.select(
            "season",
            "round",
            "driver_id",
            pl.col("position").alias("quali_position"),
        )
        if sources.qualifying.height
        else pl.DataFrame(
            schema={
                "season": pl.Int64,
                "round": pl.Int64,
                "driver_id": pl.String,
                "quali_position": pl.Int64,
            }
        )
    )

    return (
        sources.race_results.join(sprint, on=["season", "round", "driver_id"], how="left")
        .join(qualifying, on=["season", "round", "driver_id"], how="left")
        .with_columns(
            pl.lit(True).alias("entered"),
            pl.col("points").alias("race_points"),
            pl.col("sprint_points").fill_null(0.0),
            pl.col("position").alias("finish_position"),
        )
        .with_columns(
            (pl.col("race_points") + pl.col("sprint_points")).alias("round_points"),
        )
    )


def _driver_seasons(entries: pl.DataFrame) -> pl.DataFrame:
    """The first round each driver contested in each season."""
    return entries.group_by(["season", "driver_id"]).agg(pl.col("round").min().alias("first_round"))


def _build_spine(entries: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Every ``(season, round, driver_id)`` cell the dataset should contain.

    A driver enters the panel at the round of their season debut and stays in it for every
    subsequent completed round, whether or not they took part. Rows before the debut are
    not fabricated: a driver who has not yet raced is not in the championship.
    """
    completed = entries.select("season", "round").unique()
    contested = _driver_seasons(entries)

    return (
        completed.join(contested, on="season", how="inner")
        .filter(pl.col("round") >= pl.col("first_round"))
        .drop("first_round")
        .join(calendar.select("season", "round"), on=["season", "round"], how="semi")
    )


def build_panel(sources: SourceFrames) -> pl.DataFrame:
    """Build the dense driver-round panel that every feature block computes over.

    Args:
        sources: Curated Formula 1 frames.

    Returns:
        A frame keyed by ``(season, round, driver_id)``, sorted by season, driver and
        round, carrying each round's contributions plus the season context needed to
        interpret them.

    Raises:
        ValueError: If there are no race results to build a panel from.
    """
    if not sources.race_results.height:
        msg = "Cannot build a panel: no race results were loaded"
        raise ValueError(msg)

    calendar = build_calendar(sources.races)
    entries = _build_entries(sources)
    spine = _build_spine(entries, calendar)

    entry_columns = [
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "entered",
        "grid",
        "finish_position",
        "position_order",
        "race_points",
        "sprint_points",
        "round_points",
        "sprint_position",
        "quali_position",
        "finished",
        "status",
    ]
    panel = (
        spine.join(entries.select(entry_columns), on=["season", "round", "driver_id"], how="left")
        .join(
            calendar.select(
                "season",
                "round",
                pl.col("date").alias("race_date"),
                "circuit_id",
                "has_sprint",
                "round_max_points",
                "rounds_total",
                "max_points_remaining",
            ),
            on=["season", "round"],
            how="left",
        )
        .sort(["season", "driver_id", "round"])
    )

    # A driver who misses a round keeps their team: the constructor is a property of the
    # season's contract, not of the individual entry.
    panel = panel.with_columns(
        pl.col("constructor_id")
        .forward_fill()
        .backward_fill()
        .over(["season", "driver_id"])
        .alias("constructor_id"),
        pl.col("entered").fill_null(value=False),
        pl.col("finished").fill_null(value=False),
        pl.col("race_points").fill_null(0.0),
        pl.col("sprint_points").fill_null(0.0),
        pl.col("round_points").fill_null(0.0),
    )

    return panel.with_columns(
        (pl.col("finish_position") == 1).fill_null(value=False).alias("is_win"),
        (pl.col("finish_position") <= _PODIUM_POSITION).fill_null(value=False).alias("is_podium"),
        (pl.col("finish_position") <= _TOP_FIVE_POSITION)
        .fill_null(value=False)
        .alias("is_top_five"),
        (pl.col("round_points") > 0).alias("is_points_finish"),
        # Grid position is the authoritative "started from the front" signal: it survives
        # eras with no qualifying data, and reflects post-qualifying penalties.
        pl.coalesce(pl.col("quali_position") == 1, pl.col("grid") == 1)
        .fill_null(value=False)
        .alias("is_pole"),
        (pl.col("entered") & ~pl.col("finished")).alias("is_dnf"),
        pl.when(pl.col("entered"))
        .then(pl.col("grid") - pl.col("finish_position"))
        .alias("positions_gained"),
    )
