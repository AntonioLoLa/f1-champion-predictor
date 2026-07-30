"""A small, hand-consistent synthetic championship used by the feature test suite.

Two seasons, four rounds each, three drivers, two constructors -- enough to exercise every
feature block (including a sprint round and a mid-field DNF) without the size of a real
bootstrap, and entirely deterministic so leakage tests can compare exact values.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from f1cp.domain.scoring import points_system_for
from f1cp.features.context import SourceFrames

SEASONS = (2020, 2021)
ROUNDS = (1, 2, 3, 4)
DRIVERS = ("d1", "d2", "d3")
CONSTRUCTOR_OF = {"d1": "alpha", "d2": "alpha", "d3": "beta"}
CIRCUITS = ("circuit_a", "circuit_b", "circuit_c", "circuit_d")
SPRINT_ROUND = {2020: None, 2021: 2}
# One deliberate retirement, to exercise DNF / finish-rate features.
DNF = {(2021, 1, "d3")}


def _finish_order(season: int, round_number: int) -> list[str]:
    """A deterministic, round-varying finishing order for the three drivers."""
    rotation = (round_number + (season - 2020)) % len(DRIVERS)
    return DRIVERS[rotation:] + DRIVERS[:rotation]


def build_synthetic_sources() -> SourceFrames:
    """Construct a small, internally consistent :class:`SourceFrames`.

    Returns:
        Source frames spanning :data:`SEASONS`, ready to feed
        :func:`f1cp.features.pipeline.run_pipeline`.
    """
    races_rows = []
    result_rows = []
    sprint_rows = []
    quali_rows = []
    driver_standing_rows = []
    constructor_standing_out = []

    for season in SEASONS:
        system = points_system_for(season)
        cumulative_points = dict.fromkeys(DRIVERS, 0.0)
        cumulative_wins: dict[str, int] = dict.fromkeys(DRIVERS, 0)
        team_points = {"alpha": 0.0, "beta": 0.0}
        team_wins = {"alpha": 0, "beta": 0}

        for round_number in ROUNDS:
            circuit_id = CIRCUITS[(round_number - 1) % len(CIRCUITS)]
            race_date = dt.date(season, 3, 1) + dt.timedelta(days=21 * (round_number - 1))
            has_sprint = SPRINT_ROUND[season] == round_number
            races_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "name": f"{season} Round {round_number}",
                    "date": race_date,
                    "circuit_id": circuit_id,
                    "has_sprint": has_sprint,
                }
            )

            order = _finish_order(season, round_number)
            for position, driver_id in enumerate(order, start=1):
                is_dnf = (season, round_number, driver_id) in DNF
                finish_position = None if is_dnf else position
                race_points = 0.0 if is_dnf else float(system.race_points[position - 1])
                grid = ((position + 1) % len(DRIVERS)) + 1

                result_rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "driver_id": driver_id,
                        "constructor_id": CONSTRUCTOR_OF[driver_id],
                        "grid": grid,
                        "position": finish_position,
                        "position_order": position,
                        "points": race_points,
                        "laps": 0 if is_dnf else 55,
                        "status": "Retired" if is_dnf else "Finished",
                        "finished": not is_dnf,
                    }
                )
                quali_rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "driver_id": driver_id,
                        "constructor_id": CONSTRUCTOR_OF[driver_id],
                        "position": grid,
                    }
                )

                if not is_dnf:
                    cumulative_points[driver_id] += race_points
                    team_points[CONSTRUCTOR_OF[driver_id]] += race_points
                    if position == 1:
                        cumulative_wins[driver_id] += 1
                        team_wins[CONSTRUCTOR_OF[driver_id]] += 1

                if has_sprint:
                    sprint_points = (
                        float(system.sprint_points[position - 1])
                        if position <= len(system.sprint_points)
                        else 0.0
                    )
                    sprint_rows.append(
                        {
                            "season": season,
                            "round": round_number,
                            "driver_id": driver_id,
                            "constructor_id": CONSTRUCTOR_OF[driver_id],
                            "position": position,
                            "position_order": position,
                            "points": sprint_points,
                            "finished": True,
                        }
                    )
                    cumulative_points[driver_id] += sprint_points
                    team_points[CONSTRUCTOR_OF[driver_id]] += sprint_points

            ranked_drivers = sorted(DRIVERS, key=lambda d: (-cumulative_points[d], d))
            for standings_position, driver_id in enumerate(ranked_drivers, start=1):
                driver_standing_rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "driver_id": driver_id,
                        "constructor_id": CONSTRUCTOR_OF[driver_id],
                        "points": cumulative_points[driver_id],
                        "position": standings_position,
                        "wins": cumulative_wins[driver_id],
                    }
                )

            ranked_teams = sorted(team_points, key=lambda t: (-team_points[t], t))
            for team_position, constructor_id in enumerate(ranked_teams, start=1):
                constructor_standing_out.append(
                    {
                        "season": season,
                        "round": round_number,
                        "constructor_id": constructor_id,
                        "points": team_points[constructor_id],
                        "position": team_position,
                        "wins": team_wins[constructor_id],
                    }
                )

    races = pl.DataFrame(
        races_rows,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "name": pl.String,
            "date": pl.Date,
            "circuit_id": pl.String,
            "has_sprint": pl.Boolean,
        },
    )
    race_results = pl.DataFrame(
        result_rows,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "grid": pl.Int64,
            "position": pl.Int64,
            "position_order": pl.Int64,
            "points": pl.Float64,
            "laps": pl.Int64,
            "status": pl.String,
            "finished": pl.Boolean,
        },
    )
    sprint_results = pl.DataFrame(
        sprint_rows,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "position": pl.Int64,
            "position_order": pl.Int64,
            "points": pl.Float64,
            "finished": pl.Boolean,
        },
    )
    qualifying = pl.DataFrame(
        quali_rows,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "position": pl.Int64,
        },
    )
    driver_standings = pl.DataFrame(
        driver_standing_rows,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "points": pl.Float64,
            "position": pl.Int64,
            "wins": pl.Int64,
        },
    )
    constructor_standings = pl.DataFrame(
        constructor_standing_out,
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "constructor_id": pl.String,
            "points": pl.Float64,
            "position": pl.Int64,
            "wins": pl.Int64,
        },
    )
    drivers = pl.DataFrame(
        {
            "driver_id": ["d1", "d2", "d3"],
            "code": ["DR1", "DR2", "DR3"],
            "permanent_number": [1, 2, 3],
            "given_name": ["Driver", "Driver", "Driver"],
            "family_name": ["One", "Two", "Three"],
            "date_of_birth": [dt.date(1995, 1, 1), dt.date(1998, 6, 15), dt.date(2001, 11, 30)],
            "nationality": ["Dutch", "British", "Monegasque"],
        },
        schema={
            "driver_id": pl.String,
            "code": pl.String,
            "permanent_number": pl.Int64,
            "given_name": pl.String,
            "family_name": pl.String,
            "date_of_birth": pl.Date,
            "nationality": pl.String,
        },
    )
    constructors = pl.DataFrame(
        {
            "constructor_id": ["alpha", "beta"],
            "name": ["Alpha Racing", "Beta Motorsport"],
            "nationality": ["British", "Italian"],
        },
        schema={"constructor_id": pl.String, "name": pl.String, "nationality": pl.String},
    )
    circuits = pl.DataFrame(
        {
            "circuit_id": list(CIRCUITS),
            "name": [f"Circuit {letter}" for letter in "ABCD"],
            "locality": ["Town A", "Town B", "Town C", "Town D"],
            "country": ["Dutch", "British", "Monegasque", "Italian"],
            "latitude": [1.0, 2.0, 3.0, 4.0],
            "longitude": [1.0, 2.0, 3.0, 4.0],
        },
        schema={
            "circuit_id": pl.String,
            "name": pl.String,
            "locality": pl.String,
            "country": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
        },
    )

    return SourceFrames(
        races=races,
        race_results=race_results,
        sprint_results=sprint_results,
        qualifying=qualifying,
        driver_standings=driver_standings,
        constructor_standings=constructor_standings,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
    )
