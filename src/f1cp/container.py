"""Composition root.

Constructs the object graph once, in one place, so that no module reaches for global
state to find its collaborators. The CLI, the API and the tests all build an
:class:`AppContext` and pass it down; swapping the database or a provider is a change to
this file or to configuration, never to business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Self

from f1cp.config import Settings, load_settings
from f1cp.data.ingest import DataIngestor
from f1cp.data.providers.registry import ProviderRegistry
from f1cp.db.session import Database
from f1cp.inference.predictor import PredictionService
from f1cp.logging_config import configure_logging
from f1cp.training.trainer import Trainer

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["AppContext"]


@dataclass(slots=True)
class AppContext:
    """The application's wired dependencies."""

    settings: Settings
    database: Database
    registry: ProviderRegistry

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        *,
        conf_dir: Path | None = None,
        configure_logs: bool = True,
        ensure_schema: bool = False,
    ) -> Self:
        """Build the object graph.

        Args:
            settings: Pre-built settings; loaded from ``conf/`` when omitted.
            conf_dir: Configuration directory override.
            configure_logs: Whether to configure the root logger.
            ensure_schema: Create any missing tables. Convenient for local runs and tests;
                deployments should apply Alembic migrations instead.

        Returns:
            A ready-to-use context.
        """
        resolved = settings or load_settings(conf_dir)
        if configure_logs:
            configure_logging(resolved.logging.level, resolved.logging.format)
        resolved.paths.ensure_exist()

        database = Database(resolved.database)
        if ensure_schema:
            database.create_all()

        return cls(
            settings=resolved,
            database=database,
            registry=ProviderRegistry(resolved),
        )

    def ingestor(self) -> DataIngestor:
        """Build a data ingestor bound to this context."""
        return DataIngestor(self.settings, self.database, self.registry)

    def trainer(self) -> Trainer:
        """Build a model trainer bound to this context."""
        return Trainer(self.settings, self.database)

    def predictor(self) -> PredictionService:
        """Build a prediction service bound to this context.

        Callers that serve more than one request (the API) should build this once and
        reuse it, since it caches deserialised MLflow models across calls.
        """
        return PredictionService(self.settings)

    def close(self) -> None:
        """Release pooled database connections."""
        self.database.dispose()

    def __enter__(self) -> Self:
        """Enter a scope that releases database connections on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release pooled database connections."""
        self.close()
