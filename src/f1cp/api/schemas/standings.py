"""Schemas for the standings endpoint."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["DriverStandingResponse", "SeasonStandingsResponse"]


class DriverStandingResponse(BaseModel):
    """A driver's championship standing after a round."""

    driver_id: str
    driver_name: str
    constructor_id: str | None
    constructor_name: str | None
    points: float
    position: int | None
    wins: int


class SeasonStandingsResponse(BaseModel):
    """Every driver's standing after one round of a season."""

    season: int
    round: int
    standings: list[DriverStandingResponse]
