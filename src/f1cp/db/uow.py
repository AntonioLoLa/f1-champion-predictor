"""Unit of Work.

Groups the repositories behind a single transactional boundary. Ingesting a round touches
races, results, standings and the audit table; either all of those land or none do.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Self

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

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from f1cp.db.session import Database

__all__ = ["UnitOfWork"]


class UnitOfWork:
    """Transactional scope exposing every repository.

    Example:
        >>> with UnitOfWork(database) as uow:  # doctest: +SKIP
        ...     uow.races.upsert_many(rows)
        ...     uow.results.upsert_race_results(results)
        ... # committed here, or rolled back if the block raised
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._session: Session | None = None

    def __enter__(self) -> Self:
        """Open a session and bind the repositories to it."""
        self._session = self._database.session_factory()

        session = self._session
        self.circuits = CircuitRepository(session)
        self.drivers = DriverRepository(session)
        self.constructors = ConstructorRepository(session)
        self.races = RaceRepository(session)
        self.results = ResultRepository(session)
        self.standings = StandingsRepository(session)
        self.ingestions = IngestionRepository(session)
        self.datasets = DatasetVersionRepository(session)
        self.training_runs = TrainingRunRepository(session)
        self.models = ModelVersionRepository(session)
        self.evaluations = EvaluationRepository(session)
        self.predictions = PredictionRepository(session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Commit on success, roll back on error, and always close the session."""
        if self._session is None:  # pragma: no cover - defensive
            return
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            self._session.close()
            self._session = None

    @property
    def session(self) -> Session:
        """The active session.

        Returns:
            The session opened by :meth:`__enter__`.

        Raises:
            RuntimeError: If accessed outside a ``with`` block.
        """
        if self._session is None:
            msg = "UnitOfWork must be used as a context manager"
            raise RuntimeError(msg)
        return self._session

    def commit(self) -> None:
        """Commit the current transaction without leaving the block.

        Useful for long ingestion loops that should checkpoint progress per season rather
        than hold one transaction open for the whole backfill.
        """
        self.session.commit()

    def rollback(self) -> None:
        """Roll back the current transaction."""
        self.session.rollback()
