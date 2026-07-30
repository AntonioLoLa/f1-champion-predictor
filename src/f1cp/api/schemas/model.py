"""Schemas for model version endpoints."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

__all__ = ["ActivateModelResponse", "ModelVersionSummary"]


class ModelVersionSummary(BaseModel):
    """A trained model candidate, as exposed over the API."""

    id: int
    training_run_id: int
    name: str
    algorithm: str
    is_active: bool
    is_best: bool
    metrics: dict[str, float] | None
    created_at: dt.datetime


class ActivateModelResponse(BaseModel):
    """Outcome of activating a model version."""

    id: int
    name: str
    is_active: bool
