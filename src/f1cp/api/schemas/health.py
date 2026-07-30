"""Schemas for the health endpoint."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    """Service liveness and readiness summary."""

    status: str
    database: bool
    active_model: str | None
