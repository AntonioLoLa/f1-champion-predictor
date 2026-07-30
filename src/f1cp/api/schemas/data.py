"""Schemas for the data-acquisition endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["IngestionReportResponse", "UpdateDataRequest", "UpdateDataResponse"]


class UpdateDataRequest(BaseModel):
    """Options mirroring ``f1cp data update``."""

    bootstrap: bool = False
    seasons: list[int] | None = None
    refresh: bool = False
    min_season: int | None = None
    background: bool = Field(
        default=False,
        description="Run the update in the background and return immediately.",
    )


class IngestionReportResponse(BaseModel):
    """Outcome of one ingestion run."""

    provider: str
    mode: str
    seasons: list[int]
    rows_written: dict[str, int]
    status: str
    error: str | None


class UpdateDataResponse(BaseModel):
    """Outcome of an update-data request."""

    status: str
    reports: list[IngestionReportResponse]
