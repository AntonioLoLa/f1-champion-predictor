"""Schemas for the historical results archive.

Nothing here involves the model: these endpoints serve the curated Formula 1 record as it
was ingested -- what happened, not what was predicted.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

__all__ = [
    "ConstructorStandingResponse",
    "RaceResultResponse",
    "RaceSummaryResponse",
    "SeasonListResponse",
    "SeasonRacesResponse",
    "SeasonSummaryResponse",
]


class SeasonSummaryResponse(BaseModel):
    """One season available in the archive."""

    season: int
    rounds: int
    rounds_completed: int
    champion_id: str | None
    champion_name: str | None


class SeasonListResponse(BaseModel):
    """Every season with stored data, newest first."""

    seasons: list[SeasonSummaryResponse]


class RaceSummaryResponse(BaseModel):
    """One Grand Prix on a season's calendar."""

    round: int
    name: str
    date: dt.date | None
    circuit_id: str | None
    circuit_name: str | None
    country: str | None
    has_sprint: bool
    winner_id: str | None
    winner_name: str | None
    winner_constructor_id: str | None


class SeasonRacesResponse(BaseModel):
    """A season's full calendar, with the winner of each round that has been run."""

    season: int
    races: list[RaceSummaryResponse]


class RaceResultResponse(BaseModel):
    """One driver's classification in a single Grand Prix."""

    position: int | None
    position_order: int
    driver_id: str
    driver_name: str
    constructor_id: str | None
    grid: int | None
    laps: int | None
    status: str | None
    points: float
    finished: bool


class ConstructorStandingResponse(BaseModel):
    """A constructor's championship standing after a round."""

    constructor_id: str
    constructor_name: str
    points: float
    position: int | None
    wins: int
