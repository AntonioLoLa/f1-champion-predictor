"""Explainability endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database, get_predictor
from f1cp.api.schemas.explain import ExplainResponse

if TYPE_CHECKING:
    from f1cp.db.session import Database
    from f1cp.inference.predictor import PredictionService

__all__ = ["router"]

router = APIRouter(tags=["explain"])


@router.get("/explain/{season}/{round}", response_model=ExplainResponse)
def explain(
    season: int,
    round: int,
    model_version_id: int | None = None,
    database: Database = Depends(get_database),
    predictor: PredictionService = Depends(get_predictor),
) -> ExplainResponse:
    """Explain a model's inputs for the drivers contesting one round, via SHAP.

    Args:
        season: Championship year.
        round: Round within the season.
        model_version_id: Model version to explain. Defaults to the active one.
        database: The shared database handle.
        predictor: The shared prediction service.

    Returns:
        Feature importances, most important first.

    Raises:
        HTTPException: 404 if the model or the round cannot be resolved.
    """
    with database.session_scope() as uow:
        try:
            result = predictor.explain_round(
                uow, season=season, round_number=round, model_version_id=model_version_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ExplainResponse(
        season=result.season,
        round=result.round,
        model_version_id=result.model_version_id,
        feature_names=result.feature_names,
        importances=result.importances,
    )
