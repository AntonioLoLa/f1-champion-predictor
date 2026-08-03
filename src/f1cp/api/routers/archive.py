"""The historical results archive: seasons, calendars and race classifications.

Deliberately model-free. Everything served here is the curated record as ingested -- who
actually finished where -- which is what makes it the reference the prediction pages are
read against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database
from f1cp.api.schemas.archive import (
    ConstructorStandingResponse,
    PitStopResponse,
    RaceResultResponse,
    RaceSummaryResponse,
    SeasonListResponse,
    SeasonRacesResponse,
    SeasonSummaryResponse,
)

if TYPE_CHECKING:
    from f1cp.db.session import Database

__all__ = ["router"]

router = APIRouter(tags=["archive"])


@router.get("/seasons", response_model=SeasonListResponse)
def seasons(database: Database = Depends(get_database)) -> SeasonListResponse:
    """List every season with stored data, newest first.

    Args:
        database: The shared database handle.

    Returns:
        One entry per season, with its calendar length, how much of it has been run, and
        the champion where the season is complete.
    """
    with database.session_scope() as uow:
        champions = uow.standings.champion_by_season()
        names = {
            driver.driver_id: driver.full_name
            for driver in uow.drivers.get_by_ids(list(champions.values()))
        }
        summaries = []
        for season in sorted(uow.races.seasons(), reverse=True):
            champion_id = champions.get(season)
            summaries.append(
                SeasonSummaryResponse(
                    season=season,
                    rounds=uow.races.max_round(season) or 0,
                    rounds_completed=len(uow.races.rounds_with_results(season)),
                    champion_id=champion_id,
                    champion_name=names.get(champion_id) if champion_id else None,
                )
            )
        return SeasonListResponse(seasons=summaries)


@router.get("/seasons/{season}/races", response_model=SeasonRacesResponse)
def season_races(season: int, database: Database = Depends(get_database)) -> SeasonRacesResponse:
    """Return a season's calendar with the winner of every round already run.

    Args:
        season: Championship year.
        database: The shared database handle.

    Returns:
        Every round in calendar order. Rounds not yet run carry a ``None`` winner rather
        than being omitted, so the response is the full published calendar.

    Raises:
        HTTPException: 404 if the season has no stored calendar.
    """
    with database.session_scope() as uow:
        calendar = uow.races.for_season(season)
        if not calendar:
            msg = f"No calendar stored for season {season}"
            raise HTTPException(status_code=404, detail=msg)

        winners = uow.results.winners(season)
        driver_names = {
            driver.driver_id: driver.full_name
            for driver in uow.drivers.get_by_ids([winner for winner, _ in winners.values()])
        }
        circuits = {circuit.circuit_id: circuit for circuit in uow.circuits.list()}

        races = []
        for race in calendar:
            winner = winners.get(race.round)
            circuit = circuits.get(race.circuit_id) if race.circuit_id else None
            races.append(
                RaceSummaryResponse(
                    round=race.round,
                    name=race.name,
                    date=race.date,
                    circuit_id=race.circuit_id,
                    circuit_name=circuit.name if circuit else None,
                    country=circuit.country if circuit else None,
                    has_sprint=race.has_sprint,
                    winner_id=winner[0] if winner else None,
                    winner_name=driver_names.get(winner[0]) if winner else None,
                    winner_constructor_id=winner[1] if winner else None,
                )
            )
        return SeasonRacesResponse(season=season, races=races)


@router.get("/seasons/{season}/races/{round}/results", response_model=list[RaceResultResponse])
def race_results(
    season: int, round: int, database: Database = Depends(get_database)
) -> list[RaceResultResponse]:
    """Return the full classification of one Grand Prix.

    Args:
        season: Championship year.
        round: Round within the season.
        database: The shared database handle.

    Returns:
        Every classified driver, winner first, retirements last.

    Raises:
        HTTPException: 404 if that round has no stored results.
    """
    with database.session_scope() as uow:
        rows = uow.results.for_race(season, round)
        if not rows:
            msg = f"No results stored for season {season} round {round}"
            raise HTTPException(status_code=404, detail=msg)

        names = {
            driver.driver_id: driver.full_name
            for driver in uow.drivers.get_by_ids([row.driver_id for row in rows])
        }
        return [
            RaceResultResponse(
                position=row.position,
                position_order=row.position_order,
                driver_id=row.driver_id,
                driver_name=names.get(row.driver_id, row.driver_id),
                constructor_id=row.constructor_id,
                grid=row.grid,
                laps=row.laps,
                status=row.status,
                points=row.points,
                finished=row.finished,
            )
            for row in rows
        ]


@router.get("/seasons/{season}/races/{round}/pit-stops", response_model=list[PitStopResponse])
def pit_stops(
    season: int, round: int, database: Database = Depends(get_database)
) -> list[PitStopResponse]:
    """Return every pit stop of one Grand Prix.

    Args:
        season: Championship year.
        round: Round within the season.
        database: The shared database handle.

    Returns:
        Pit stops in driver, then stop-number order. Empty rather than 404 for a race run
        before 2011 or with no in-race pit stop (rare, but the source data allows it) --
        only an unrun or unknown round is a 404.

    Raises:
        HTTPException: 404 if that round has no stored results.
    """
    with database.session_scope() as uow:
        if not uow.results.for_race(season, round):
            msg = f"No results stored for season {season} round {round}"
            raise HTTPException(status_code=404, detail=msg)

        rows = uow.results.pit_stops_for_race(season, round)
        names = {
            driver.driver_id: driver.full_name
            for driver in uow.drivers.get_by_ids([row.driver_id for row in rows])
        }
        return [
            PitStopResponse(
                driver_id=row.driver_id,
                driver_name=names.get(row.driver_id, row.driver_id),
                stop=row.stop,
                lap=row.lap,
                duration_ms=row.duration_ms,
            )
            for row in rows
        ]


@router.get(
    "/seasons/{season}/constructor-standings", response_model=list[ConstructorStandingResponse]
)
def constructor_standings(
    season: int, round: int | None = None, database: Database = Depends(get_database)
) -> list[ConstructorStandingResponse]:
    """Return the constructors' championship standings after one round.

    Args:
        season: Championship year.
        round: Round within the season. Defaults to the latest round with stored results.
        database: The shared database handle.

    Returns:
        Every constructor's points, position and wins after that round.

    Raises:
        HTTPException: 404 if the season or round has no stored standings.
    """
    with database.session_scope() as uow:
        resolved_round = round
        if resolved_round is None:
            rounds = uow.races.rounds_with_results(season)
            if not rounds:
                msg = f"No results stored for season {season}"
                raise HTTPException(status_code=404, detail=msg)
            resolved_round = max(rounds)

        rows = uow.standings.constructors_for_round(season, resolved_round)
        if not rows:
            msg = f"No constructor standings for season {season} round {resolved_round}"
            raise HTTPException(status_code=404, detail=msg)

        names = {
            constructor.constructor_id: constructor.name for constructor in uow.constructors.list()
        }
        return [
            ConstructorStandingResponse(
                constructor_id=row.constructor_id,
                constructor_name=names.get(row.constructor_id, row.constructor_id),
                points=row.points,
                position=row.position,
                wins=row.wins,
            )
            for row in rows
        ]
