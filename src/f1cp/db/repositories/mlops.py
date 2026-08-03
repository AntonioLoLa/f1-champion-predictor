"""Repositories for MLOps metadata: ingestions, datasets, runs, models and predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from f1cp.db.models import (
    DataIngestion,
    DatasetVersion,
    Evaluation,
    ModelVersion,
    Prediction,
    TrainingRun,
)
from f1cp.db.repositories.base import SqlAlchemyRepository, bulk_upsert

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DatasetVersionRepository",
    "EvaluationRepository",
    "IngestionRepository",
    "ModelVersionRepository",
    "PredictionRepository",
    "TrainingRunRepository",
]


class IngestionRepository(SqlAlchemyRepository[DataIngestion]):
    """Audit trail of data acquisition runs."""

    model = DataIngestion

    def recent(self, limit: int = 20) -> Sequence[DataIngestion]:
        """Most recent ingestion runs, newest first.

        Args:
            limit: Maximum number of records.

        Returns:
            Ingestion audit records.
        """
        statement = select(DataIngestion).order_by(DataIngestion.started_at.desc()).limit(limit)
        return self.session.execute(statement).scalars().all()


class DatasetVersionRepository(SqlAlchemyRepository[DatasetVersion]):
    """Built feature matrices."""

    model = DatasetVersion

    def get_by_version(self, version: str) -> DatasetVersion | None:
        """Look up a dataset by its version string.

        Args:
            version: The dataset version identifier.

        Returns:
            The matching record, or ``None``.
        """
        statement = select(DatasetVersion).where(DatasetVersion.version == version)
        return self.session.execute(statement).scalars().first()

    def latest(self) -> DatasetVersion | None:
        """The most recently built dataset."""
        statement = select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1)
        return self.session.execute(statement).scalars().first()


class TrainingRunRepository(SqlAlchemyRepository[TrainingRun]):
    """Training pipeline executions."""

    model = TrainingRun

    def get_by_uuid(self, run_uuid: str) -> TrainingRun | None:
        """Look up a run by its external UUID.

        The API hands this UUID back to callers so they can poll a background training job
        without exposing database primary keys.

        Args:
            run_uuid: The run's UUID.

        Returns:
            The matching run, or ``None``.
        """
        statement = select(TrainingRun).where(TrainingRun.run_uuid == run_uuid)
        return self.session.execute(statement).scalars().first()

    def recent(self, limit: int = 20) -> Sequence[TrainingRun]:
        """Most recent training runs, newest first.

        Args:
            limit: Maximum number of records.

        Returns:
            Training run records.
        """
        statement = select(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(limit)
        return self.session.execute(statement).scalars().all()


class ModelVersionRepository(SqlAlchemyRepository[ModelVersion]):
    """Trained model candidates and the currently served model."""

    model = ModelVersion

    def get_active(self) -> ModelVersion | None:
        """The model currently serving predictions.

        Returns:
            The active model version, or ``None`` if nothing has been activated.
        """
        statement = select(ModelVersion).where(ModelVersion.is_active.is_(True))
        return self.session.execute(statement).scalars().first()

    def list_versions(self, limit: int = 50) -> Sequence[ModelVersion]:
        """All model versions, newest first.

        Args:
            limit: Maximum number of records.

        Returns:
            Model version records.
        """
        statement = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        return self.session.execute(statement).scalars().all()

    def for_training_run(self, training_run_id: int) -> Sequence[ModelVersion]:
        """Every candidate produced by one training run.

        Used to look up which candidate a background training run selected as its winner
        once the run finishes, without the caller needing to know model version ids in
        advance.

        Args:
            training_run_id: Primary key of the training run.

        Returns:
            Model version records, in no particular order.
        """
        statement = select(ModelVersion).where(ModelVersion.training_run_id == training_run_id)
        return self.session.execute(statement).scalars().all()

    def activate(self, model_version_id: int) -> ModelVersion:
        """Make one model version the active one, deactivating any other.

        Done in two statements inside the caller's transaction so the database is never
        observably in a state with two active models.

        Args:
            model_version_id: Primary key of the version to activate.

        Returns:
            The now-active model version.

        Raises:
            ValueError: If no such model version exists.

        Note:
            This does not check ``mlflow_model_uri`` -- the trainer activates its own
            selected winner through this same method right after training, and that must
            keep succeeding even when MLflow logging is disabled or degrades non-strictly
            (see :class:`~f1cp.training.tracking.ExperimentTracker`). The API endpoint that
            lets an operator activate an arbitrary version is where servability should be
            checked instead; see :func:`f1cp.api.routers.model.activate`.
        """
        target = self.get(model_version_id)
        if target is None:
            msg = f"Model version {model_version_id} does not exist"
            raise ValueError(msg)

        self.session.execute(
            update(ModelVersion).where(ModelVersion.is_active.is_(True)).values(is_active=False)
        )
        self.session.execute(
            update(ModelVersion).where(ModelVersion.id == model_version_id).values(is_active=True)
        )
        self.session.flush()
        self.session.refresh(target)
        return target


class EvaluationRepository(SqlAlchemyRepository[Evaluation]):
    """Metric values in long format."""

    model = Evaluation

    def add_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert metric rows.

        Args:
            rows: Dictionaries matching the :class:`~f1cp.db.models.Evaluation` columns.

        Returns:
            Number of rows written.
        """
        materialised = list(rows)
        if materialised:
            self.session.bulk_insert_mappings(Evaluation, materialised)
        return len(materialised)

    def for_model(
        self,
        model_version_id: int,
        *,
        split: str | None = None,
        phase: str | None = None,
    ) -> Sequence[Evaluation]:
        """Metrics recorded for one model version.

        Args:
            model_version_id: The model version to filter by.
            split: Optional split filter (``train``/``validation``/``test``).
            phase: Optional season-phase filter.

        Returns:
            Matching metric rows.
        """
        statement = select(Evaluation).where(Evaluation.model_version_id == model_version_id)
        if split is not None:
            statement = statement.where(Evaluation.split == split)
        if phase is not None:
            statement = statement.where(Evaluation.phase == phase)
        return self.session.execute(statement).scalars().all()


class PredictionRepository(SqlAlchemyRepository[Prediction]):
    """Stored championship probabilities."""

    model = Prediction

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update predictions.

        Re-predicting a round with the same model must overwrite rather than duplicate,
        hence the unique key on ``(model_version_id, season, round, driver_id)``.

        Args:
            rows: Prediction dictionaries.

        Returns:
            Number of rows written.
        """
        return bulk_upsert(
            self.session,
            Prediction,
            rows,
            index_elements=["model_version_id", "season", "round", "driver_id"],
        )

    def for_round(
        self,
        season: int,
        round_number: int,
        *,
        model_version_id: int | None = None,
    ) -> Sequence[Prediction]:
        """Predictions for one round, best-ranked first.

        Args:
            season: Championship year.
            round_number: Round within the season.
            model_version_id: Restrict to one model version, or ``None`` for all.

        Returns:
            Prediction rows ordered by rank.
        """
        statement = select(Prediction).where(
            Prediction.season == season, Prediction.round == round_number
        )
        if model_version_id is not None:
            statement = statement.where(Prediction.model_version_id == model_version_id)
        return self.session.execute(statement.order_by(Prediction.rank)).scalars().all()

    def for_season(
        self,
        season: int,
        *,
        model_version_id: int | None = None,
    ) -> Sequence[Prediction]:
        """Every stored prediction for a season, ordered by round then rank.

        Args:
            season: Championship year.
            model_version_id: Restrict to one model version, or ``None`` for all.

        Returns:
            Prediction rows.
        """
        statement = select(Prediction).where(Prediction.season == season)
        if model_version_id is not None:
            statement = statement.where(Prediction.model_version_id == model_version_id)
        return (
            self.session.execute(statement.order_by(Prediction.round, Prediction.rank))
            .scalars()
            .all()
        )

    def for_driver(self, season: int, driver_id: str) -> Sequence[Prediction]:
        """A single driver's probability trajectory across a season.

        Args:
            season: Championship year.
            driver_id: Driver identifier.

        Returns:
            Prediction rows ordered by round.
        """
        statement = (
            select(Prediction)
            .where(Prediction.season == season, Prediction.driver_id == driver_id)
            .order_by(Prediction.round)
        )
        return self.session.execute(statement).scalars().all()
