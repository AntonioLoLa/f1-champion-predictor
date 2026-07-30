"""Repository implementations."""

from f1cp.db.repositories.base import AbstractRepository, SqlAlchemyRepository, bulk_upsert
from f1cp.db.repositories.f1_data import (
    CircuitRepository,
    ConstructorRepository,
    DriverRepository,
    RaceRepository,
    ResultRepository,
    StandingsRepository,
)
from f1cp.db.repositories.mlops import (
    DatasetVersionRepository,
    EvaluationRepository,
    IngestionRepository,
    ModelVersionRepository,
    PredictionRepository,
    TrainingRunRepository,
)

__all__ = [
    "AbstractRepository",
    "CircuitRepository",
    "ConstructorRepository",
    "DatasetVersionRepository",
    "DriverRepository",
    "EvaluationRepository",
    "IngestionRepository",
    "ModelVersionRepository",
    "PredictionRepository",
    "RaceRepository",
    "ResultRepository",
    "SqlAlchemyRepository",
    "StandingsRepository",
    "TrainingRunRepository",
    "bulk_upsert",
]
