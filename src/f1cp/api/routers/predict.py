"""Championship-probability prediction endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database, get_predictor
from f1cp.api.schemas.predict import (
    DriverPredictionResponse,
    PredictRequest,
    RoundPredictionsResponse,
)

if TYPE_CHECKING:
    from f1cp.db.session import Database
    from f1cp.inference.predictor import PredictionService, RoundPredictions

__all__ = ["router"]

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=RoundPredictionsResponse)
def predict(
    payload: PredictRequest,
    database: Database = Depends(get_database),
    predictor: PredictionService = Depends(get_predictor),
) -> RoundPredictionsResponse:
    """Score every driver contesting one round.

    Args:
        payload: Which round and model to score; defaults to the latest round with stored
            results and the currently active model.
        database: The shared database handle.
        predictor: The shared prediction service.

    Returns:
        Ranked championship probabilities for that round.

    Raises:
        HTTPException: 404 if the model or the requested round cannot be resolved.
    """
    with database.session_scope() as uow:
        try:
            result = predictor.predict_round(
                uow,
                season=payload.season,
                round_number=payload.round,
                model_version_id=payload.model_version_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(result)


@router.get("/predict-season/{season}", response_model=list[RoundPredictionsResponse])
def predict_season(
    season: int,
    model_version_id: int | None = None,
    database: Database = Depends(get_database),
    predictor: PredictionService = Depends(get_predictor),
) -> list[RoundPredictionsResponse]:
    """Score every round of a season that already has stored results.

    Args:
        season: Championship year.
        model_version_id: Model version to score with. Defaults to the active one.
        database: The shared database handle.
        predictor: The shared prediction service.

    Returns:
        One set of ranked probabilities per round, in round order.

    Raises:
        HTTPException: 404 if the model or the season cannot be resolved.
    """
    with database.session_scope() as uow:
        try:
            results = predictor.predict_season(uow, season, model_version_id=model_version_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [_to_response(result) for result in results]


def _to_response(result: RoundPredictions) -> RoundPredictionsResponse:
    """Project a :class:`~f1cp.inference.predictor.RoundPredictions` onto its API response."""
    return RoundPredictionsResponse(
        season=result.season,
        round=result.round,
        model_version_id=result.model_version_id,
        model_name=result.model_name,
        drivers=[
            DriverPredictionResponse(
                driver_id=driver.driver_id,
                driver_name=driver.driver_name,
                constructor_id=driver.constructor_id,
                probability=driver.probability,
                probability_calibrated=driver.probability_calibrated,
                probability_raw=driver.probability_raw,
                rank=driver.rank,
            )
            for driver in result.drivers
        ],
    )
