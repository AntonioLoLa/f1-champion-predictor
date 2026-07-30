"""MLflow experiment tracking.

A thin wrapper rather than calling the MLflow SDK directly from the trainer: it centralises
the "is tracking even enabled" branch and the "strict" failure mode (config-gated: a broken
tracking server should abort a CI training run, but never block a developer's laptop), and
it is the one seam training tests replace with a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sklearn.pipeline import Pipeline

    from f1cp.config.settings import MLflowConfig

__all__ = ["ExperimentTracker"]

logger = get_logger(__name__)


class ExperimentTracker:
    """Records params, metrics and model artifacts to MLflow, or does nothing.

    Every method degrades to a no-op when tracking is disabled, so callers never need an
    ``if settings.mlflow.enabled`` branch of their own.
    """

    def __init__(self, config: MLflowConfig, *, artifact_location: str | None = None) -> None:
        """Configure the tracker.

        Args:
            config: MLflow settings, including whether tracking is enabled at all.
            artifact_location: Where a *new* experiment's artifacts are stored (a
                ``file://`` URI), used only the first time this experiment is created.
                Without it, a direct (non-server) sqlite/file tracking store falls back to
                MLflow's own default -- a path resolved relative to whatever the current
                working directory happens to be at that moment, then baked permanently
                into the experiment. That default is what :class:`~f1cp.container.
                AppContext` avoids by passing ``settings.paths.artifacts`` here: a location
                anchored the same way every other configured path is, so it stays valid
                regardless of which directory a command is run from.
        """
        self._config = config
        self._artifact_location = artifact_location
        self._active = False
        if config.enabled:
            self._configure()

    def _configure(self) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self._config.tracking_uri)
            self._ensure_experiment()
            self._active = True
        except Exception:
            logger.exception("Failed to configure MLflow tracking")
            if self._config.strict:
                raise

    def _ensure_experiment(self) -> None:
        """Select the configured experiment, creating it with an explicit location if new.

        MLflow only lets ``artifact_location`` be set at creation time, so an experiment
        that already exists (including one created before this parameter existed) keeps
        whatever location it was first given -- this only affects experiments created from
        here on.
        """
        import mlflow

        existing = mlflow.get_experiment_by_name(self._config.experiment_name)
        if existing is None and self._artifact_location is not None:
            mlflow.create_experiment(
                self._config.experiment_name, artifact_location=self._artifact_location
            )
        mlflow.set_experiment(self._config.experiment_name)

    @contextmanager
    def run(self, name: str, *, tags: dict[str, str] | None = None) -> Iterator[str | None]:
        """Open an MLflow run for the duration of the ``with`` block.

        Args:
            name: Run name, shown in the MLflow UI.
            tags: Optional key/value tags (e.g. ``{"fold": "3", "model": "xgboost"}``).

        Yields:
            The MLflow run id, or ``None`` if tracking is disabled.
        """
        if not self._active:
            yield None
            return

        import mlflow

        try:
            with mlflow.start_run(run_name=name, tags=tags or {}) as active_run:
                yield active_run.info.run_id
        except Exception:
            logger.exception("MLflow run failed", extra={"run_name": name})
            if self._config.strict:
                raise
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters, skipped silently when tracking is disabled."""
        if not self._active:
            return
        import mlflow

        mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics, skipped silently when tracking is disabled."""
        if not self._active:
            return
        import mlflow

        mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})

    def log_model(self, pipeline: Pipeline, name: str) -> str | None:
        """Log a fitted pipeline as an MLflow model artifact.

        Args:
            pipeline: The fitted (and, typically, calibrated) model.
            name: The logged model's name. MLflow 3 tracks every logged model as a named
                entity, so this must be a plain identifier -- no ``/``, ``:``, ``.``, ``%``
                or quote characters.

        Returns:
            The ``runs:/...`` URI of the logged model, or ``None`` if tracking is disabled.
        """
        if not self._active:
            return None
        import mlflow.sklearn

        # MLflow 3 defaults to "skops" serialization, which whitelists which classes it
        # will deserialize and rejects sklearn-internal ones our pipelines actually use
        # (e.g. `CalibratedClassifierCV`'s private `_CalibratedClassifier`). "cloudpickle"
        # serializes the object graph as-is and is standard practice for MLflow model
        # artifacts that are only ever loaded back by this same trusted deployment.
        info = mlflow.sklearn.log_model(pipeline, name=name, serialization_format="cloudpickle")
        return str(info.model_uri)

    def register_model(self, model_uri: str, name: str) -> str | None:
        """Register a logged model in the MLflow Model Registry.

        Args:
            model_uri: URI returned by :meth:`log_model`.
            name: Registered model name.

        Returns:
            The registered version number as a string, or ``None`` if tracking is disabled
            or registration failed non-strictly.
        """
        if not self._active:
            return None
        import mlflow

        try:
            result = mlflow.register_model(model_uri, name)
            return str(result.version)
        except Exception:
            logger.exception("Failed to register model with MLflow")
            if self._config.strict:
                raise
            return None
