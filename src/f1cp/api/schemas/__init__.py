"""Pydantic request/response models for the FastAPI service."""

from f1cp.api.schemas.archive import (
    ConstructorStandingResponse,
    RaceResultResponse,
    RaceSummaryResponse,
    SeasonListResponse,
    SeasonRacesResponse,
    SeasonSummaryResponse,
)
from f1cp.api.schemas.contention import DriverContentionResponse, TitleContentionResponse
from f1cp.api.schemas.data import IngestionReportResponse, UpdateDataRequest, UpdateDataResponse
from f1cp.api.schemas.explain import ExplainResponse
from f1cp.api.schemas.health import HealthResponse
from f1cp.api.schemas.model import ActivateModelResponse, ModelVersionSummary
from f1cp.api.schemas.predict import (
    DriverPredictionResponse,
    PredictRequest,
    RoundPredictionsResponse,
)
from f1cp.api.schemas.standings import DriverStandingResponse, SeasonStandingsResponse
from f1cp.api.schemas.train import TrainRequest, TrainResponse, TrainStatusResponse

__all__ = [
    "ActivateModelResponse",
    "ConstructorStandingResponse",
    "DriverContentionResponse",
    "DriverPredictionResponse",
    "DriverStandingResponse",
    "ExplainResponse",
    "HealthResponse",
    "IngestionReportResponse",
    "ModelVersionSummary",
    "PredictRequest",
    "RaceResultResponse",
    "RaceSummaryResponse",
    "RoundPredictionsResponse",
    "SeasonListResponse",
    "SeasonRacesResponse",
    "SeasonSummaryResponse",
    "SeasonStandingsResponse",
    "TitleContentionResponse",
    "TrainRequest",
    "TrainResponse",
    "TrainStatusResponse",
    "UpdateDataRequest",
    "UpdateDataResponse",
]
