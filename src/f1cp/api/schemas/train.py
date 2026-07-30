"""Schemas for the training endpoints."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

__all__ = ["TrainRequest", "TrainResponse", "TrainStatusResponse"]


class TrainRequest(BaseModel):
    """Options for scheduling a training run."""

    dataset_version: str | None = None


class TrainResponse(BaseModel):
    """Acknowledgement that a training run has been scheduled."""

    run_uuid: str
    status: str


class TrainStatusResponse(BaseModel):
    """Current state of a training run."""

    run_uuid: str
    status: str
    started_at: dt.datetime
    finished_at: dt.datetime | None
    error: str | None
    best_model_version_id: int | None
