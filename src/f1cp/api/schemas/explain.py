"""Schemas for the explainability endpoint."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["ExplainResponse"]


class ExplainResponse(BaseModel):
    """Feature attribution for a model's predictions at one round."""

    season: int
    round: int
    model_version_id: int
    feature_names: list[str]
    importances: list[float]
