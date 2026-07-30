"""Schemas for the title-contention endpoint."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["DriverContentionResponse", "TitleContentionResponse"]


class DriverContentionResponse(BaseModel):
    """Whether one driver can still mathematically win the championship."""

    driver_id: str
    driver_name: str
    current_points: float
    best_case_points: float
    still_alive: bool
    controls_own_destiny: bool
    eliminated_by: str | None
    eliminated_by_name: str | None


class TitleContentionResponse(BaseModel):
    """Title-contention status for every driver with a standing at one round."""

    season: int
    round: int
    contenders: list[DriverContentionResponse]
