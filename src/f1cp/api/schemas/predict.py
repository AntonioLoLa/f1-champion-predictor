"""Schemas for the prediction endpoints."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["DriverPredictionResponse", "PredictRequest", "RoundPredictionsResponse"]


class PredictRequest(BaseModel):
    """Options for a single-round prediction.

    Season and round both default to the latest race with stored results; the model
    defaults to the currently active version.
    """

    season: int | None = None
    round: int | None = None
    model_version_id: int | None = None


class DriverPredictionResponse(BaseModel):
    """One driver's championship probability after a round."""

    driver_id: str
    driver_name: str
    constructor_id: str | None
    probability: float
    probability_calibrated: float
    probability_raw: float
    rank: int


class RoundPredictionsResponse(BaseModel):
    """Every driver's ranked probability for one round."""

    season: int
    round: int
    model_version_id: int
    model_name: str
    drivers: list[DriverPredictionResponse]
