"""A small, hand-consistent synthetic championship used by the feature test suite.

Three seasons, six rounds, four drivers, two constructors. Every element beyond the bare
minimum is here because a real defect slipped past the earlier, tidier version of this
fixture -- where three drivers contested every round of every season, no circuit repeated,
and nobody ever arrived or missed a race:

* **A circuit raced twice in one season** (``circuit_a`` at rounds 1 and 4). The circuit
  block used to join its history on ``(season, round, driver, circuit)``, which only ever
  matched in exactly this situation -- so against real data 99.7% of the feature was null
  and against this fixture it was never exercised at all.
* **A mid-season debutant** (``d4``, from round 3 of the final season). Season rate
  features divided by rounds *elapsed* rather than rounds *started*, which is invisible
  unless somebody joins late.
* **A driver missing a round** (``d2``, round 3 of the middle season). A driver who sits
  out is still in the championship and must still get a panel row; dropping those rows is
  what once let a round's probabilities be rescaled without the eventual champion in them.
* **A retirement** (``d3``, round 1 of the final season) and **a sprint round**, both
  carried over from the original fixture.

Every value is deterministic so the leakage test can compare exact numbers.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from f1cp.domain.scoring import points_system_for
from f1cp.features.context import SourceFrames

SEASONS = (2019, 2020, 2021)
ROUNDS = (1, 2, 3, 4, 5, 6)
DRIVERS = ("d1", "d2", "d3", "d4")
CONSTRUCTOR_OF = {"d1": "alpha", "d2": "alpha", "d3": "beta", "d4": "beta"}

# `circuit_a` and `circuit_b` are each raced twice a season: enough for a driver to build a
# per-circuit history, and the case the circuit block's join used to depend on.
CIRCUIT_OF_ROUND = {
    1: "circuit_a",
    2: "circuit_b",
    3: "circuit_c",
    4: "circuit_a",
    5: "circuit_d",
    6: "circuit_b",
}
CIRCUITS = ("circuit_a", "circuit_b", "circuit_c", "circuit_d")

SPRINT_ROUND = {2019: None, 2020: None, 2021: 2}

#: ``d4`` is a mid-season arrival in the last season and never appears before that.
DEBUT = {"d1": (2019, 1), "d2": (2019, 1), "d3": (2019, 1), "d4": (2021, 3)}

#: A round a driver sat out while still contesting the championship.
ABSENCES = {(2020, 3, "d2")}

#: One deliberate retirement, to exercise DNF / rate features. Lands on whoever this
#: round's rotation puts in P1, so that round also has no classified winner.
DNF = {(2021, 1, "d1")}


def entrants(season: int, round_number: int) -> tuple[str, ...]:
    """Drivers who took part in one round, in classification order.

    Args:
        season: Championship year.
        round_number: Round within the season.

    Returns:
        The driver ids that started, ordered by finishing position.
    """
    started = [
        driver_id
        for driver_id in DRIVERS
        if DEBUT[driver_id] <= (season, round_number)
        and (season, round_number, driver_id) not in ABSENCES
    ]
    rotation = (round_number + (season - SEASONS[0])) % len(started)
    return tuple(started[rotation:] + started[:rotation])


def expected_panel_rows() -> int:
    """How many ``(season, round, driver)`` rows the panel should hold.

    A driver joins the panel at their season debut and stays for every later round,
    whether or not they started it -- so this counts rounds from debut onwards, not
    rounds actually contested.
    """
    total = 0
    for season in SEASONS:
        for driver_id in DRIVERS:
            debut_season, debut_round = DEBUT[driver_id]
            if debut_season > season:
                continue
            first = debut_round if debut_season == season else ROUNDS[0]
            total += len([r for r in ROUNDS if r >= first])
    return total


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
        cumulative_points: dict[str, float] = {}
        cumulative_wins: dict[str, int] = {}
        team_points = {"alpha": 0.0, "beta": 0.0}
        team_wins = {"alpha": 0, "beta": 0}

        for round_number in ROUNDS:
            circuit_id = CIRCUIT_OF_ROUND[round_number]
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

            order = entrants(season, round_number)
            for position, driver_id in enumerate(order, start=1):
                cumulative_points.setdefault(driver_id, 0.0)
                cumulative_wins.setdefault(driver_id, 0)

                is_dnf = (season, round_number, driver_id) in DNF
                finish_position = None if is_dnf else position
                race_points = 0.0 if is_dnf else float(system.race_points[position - 1])
                grid = ((position + 1) % len(order)) + 1

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

            # Standings list everyone who has entered the championship so far this season,
            # including anyone sitting this round out.
            ranked_drivers = sorted(cumulative_points, key=lambda d: (-cumulative_points[d], d))
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
            "driver_id": list(DRIVERS),
            "code": ["DR1", "DR2", "DR3", "DR4"],
            "permanent_number": [1, 2, 3, 4],
            "given_name": ["Driver", "Driver", "Driver", "Driver"],
            "family_name": ["One", "Two", "Three", "Four"],
            "date_of_birth": [
                dt.date(1995, 1, 1),
                dt.date(1998, 6, 15),
                dt.date(2001, 11, 30),
                dt.date(2003, 4, 9),
            ],
            "nationality": ["Dutch", "British", "Monegasque", "Italian"],
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
