"""Training orchestration.

For every enabled model-zoo candidate: walk forward over completed seasons, fitting on a
prefix of seasons, calibrating on the final few seasons of that prefix, and scoring on the
season(s) immediately after (see :mod:`f1cp.training.splitters`). Once every fold is
scored, each candidate is refit once more on (almost) everything known -- the model that
selection picks is never the one evaluated in cross-validation, only the one trained the
same way on more data, exactly as it would be retrained in production.

The database write is split from the fitting: a ``TrainingRun`` row is inserted and
committed as ``RUNNING`` before any model is fit (so a crash mid-run is visible rather than
leaving no trace), the fitting and scoring themselves touch no database, and the final
write -- every ``ModelVersion``, its ``Evaluation`` rows, activation of the winner, and
marking the run ``SUCCEEDED`` -- happens as one closing transaction. A failure in between
is recorded against the same row as ``FAILED`` in its own transaction before the original
exception is re-raised, since :class:`~f1cp.db.uow.UnitOfWork` rolls back whatever was
pending in a failed ``with`` block.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from f1cp.config import PROJECT_ROOT
from f1cp.db.models import ModelVersion, TrainingRun
from f1cp.domain.enums import RunStatus
from f1cp.features.context import KEY_COLUMNS
from f1cp.logging_config import get_logger
from f1cp.models.calibration import calibrate
from f1cp.models.normalizer import GroupProbabilityNormalizer
from f1cp.models.zoo import build_zoo
from f1cp.training.metrics import compute_metrics, final_round_top1_hit_rate
from f1cp.training.selection import select_best
from f1cp.training.splitters import RollingWindowSplit, SeasonForwardSplit, completed_seasons
from f1cp.training.tracking import ExperimentTracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.pipeline import Pipeline

    from f1cp.config.settings import Settings, ValidationConfig
    from f1cp.db.models import DatasetVersion
    from f1cp.db.session import Database
    from f1cp.db.uow import UnitOfWork
    from f1cp.models.base import ModelSpec
    from f1cp.training.splitters import Fold, Splitter

__all__ = ["CandidateResult", "Trainer", "TrainingResult"]

logger = get_logger(__name__)

_PROBABILITY_COLUMN = "probability"
_NORMALIZED_COLUMN = f"{_PROBABILITY_COLUMN}_normalized"
_SEASON_COLUMN = "season"


@dataclass(slots=True)
class CandidateResult:
    """One model zoo candidate's outcome for a training run.

    Attributes:
        name: Model zoo entry name, e.g. ``"xgboost"``.
        estimator_name: Underlying algorithm identifier.
        params: Hyperparameters used to build the estimator.
        metrics: Pooled metrics, averaged across CV folds. Used for selection.
        phase_metrics: Per-season-phase metrics, averaged across CV folds.
        fold_metrics: Pooled metrics for each individual fold, in fold order.
        final_pipeline: The fitted (and, if enabled, calibrated) pipeline refit on
            (almost) every completed season -- the artifact that gets deployed.
        base_pipeline: The same refit, *before* calibration. Calibration only rescales
            the output probability monotonically, so this is what explainability (SHAP,
            permutation importance) should attribute feature importance against, rather
            than reaching into a ``CalibratedClassifierCV``'s private internals.
        oof_predictions_path: Where the concatenated, leakage-free out-of-fold
            predictions from every CV fold were written, for evaluation to score without
            re-fitting.
        mlflow_run_id: The MLflow run id this candidate was logged under, if tracking is
            enabled.
        mlflow_model_uri: The ``models:/...`` URI of the logged calibrated model, if
            tracking is enabled.
        mlflow_base_model_uri: The URI of the logged pre-calibration pipeline, if
            tracking is enabled.
    """

    name: str
    estimator_name: str
    params: dict[str, object]
    metrics: dict[str, float | None]
    phase_metrics: dict[str, dict[str, float | None]]
    fold_metrics: list[dict[str, float | None]]
    final_pipeline: Pipeline
    base_pipeline: Pipeline
    oof_predictions_path: str
    mlflow_run_id: str | None = None
    mlflow_model_uri: str | None = None
    mlflow_base_model_uri: str | None = None


@dataclass(slots=True)
class TrainingResult:
    """Outcome of one :meth:`Trainer.run` call.

    Attributes:
        run_uuid: External identifier for this training run.
        training_run_id: Primary key of the persisted ``TrainingRun`` row.
        candidates: Every trained candidate, in model-zoo order.
        best_model_name: Name of the selected winner.
        best_model_version_id: Primary key of the winner's ``ModelVersion`` row, now
            active.
    """

    run_uuid: str
    training_run_id: int
    candidates: list[CandidateResult]
    best_model_name: str
    best_model_version_id: int


@dataclass(frozen=True, slots=True)
class _ScoredFold:
    """One fold's metrics and row-level predictions for one candidate."""

    pooled_metrics: dict[str, float | None]
    phase_metrics: dict[str, dict[str, float | None]]
    frame: pl.DataFrame
    """The fold's test rows, phase-tagged, with calibrated and normalized probabilities."""


class Trainer:
    """Fits, calibrates, scores, selects and persists every enabled model-zoo candidate."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            settings: Application settings.
            database: Database to read the dataset registry from and write results to.
            tracker: Experiment tracker. Built from ``settings.mlflow`` when omitted.
        """
        self._settings = settings
        self._database = database
        self._tracker = tracker or ExperimentTracker(
            settings.mlflow, artifact_location=f"file://{settings.paths.artifacts}"
        )

    def run(
        self, *, dataset_version: str | None = None, run_uuid: str | None = None
    ) -> TrainingResult:
        """Run one full training pass: CV score, refit, select, persist, activate.

        Args:
            dataset_version: Explicit dataset version to train on. Defaults to the most
                recently built one.
            run_uuid: External identifier to record the run under. Generated when omitted;
                callers that need to know the id before the run finishes (the API schedules
                training in the background and hands this id back immediately so a client
                can poll it) generate it themselves and pass it in.

        Returns:
            A summary of every candidate and the selected, now-active winner.

        Raises:
            ValueError: If no matching dataset exists, there is not enough season history
                to form a single walk-forward fold, or no model zoo entry is enabled.
        """
        run_uuid = run_uuid or str(uuid.uuid4())

        with self._database.session_scope() as uow:
            version = self._resolve_dataset_version(uow, dataset_version)
            training_run = TrainingRun(
                run_uuid=run_uuid,
                dataset_version_id=version.id,
                status=str(RunStatus.RUNNING),
                git_sha=_git_sha(),
                config_hash=_hash_training_config(self._settings),
                params={"dataset_version": version.version},
            )
            uow.training_runs.add(training_run)
            uow.commit()
            training_run_id = training_run.id
            dataset_path = version.path
            feature_names = list(version.feature_names)

        try:
            frame = pl.read_parquet(dataset_path)
            candidates = self._train_all_candidates(frame, feature_names, run_uuid)
        except Exception as exc:
            self._mark_failed(training_run_id, exc)
            raise

        with self._database.session_scope() as uow:
            model_versions_by_name: dict[str, ModelVersion] = {}
            for candidate in candidates:
                model_version = ModelVersion(
                    training_run_id=training_run_id,
                    name=candidate.name,
                    algorithm=candidate.estimator_name,
                    mlflow_model_uri=candidate.mlflow_model_uri,
                    mlflow_base_model_uri=candidate.mlflow_base_model_uri,
                    oof_predictions_path=candidate.oof_predictions_path,
                    artifact_path=f"models/{candidate.name}",
                    metrics={k: v for k, v in candidate.metrics.items() if v is not None},
                    params=candidate.params,
                    feature_names=feature_names,
                )
                uow.models.add(model_version)
                uow.session.flush()
                uow.evaluations.add_many(_evaluation_rows(model_version.id, candidate))
                model_versions_by_name[candidate.name] = model_version

            best_candidate = select_best(candidates, self._settings.training.selection)
            best_version = model_versions_by_name[best_candidate.name]
            best_version.is_best = True
            uow.models.activate(best_version.id)

            run = uow.training_runs.get(training_run_id)
            if run is None:  # pragma: no cover - defensive, row was just created
                msg = f"Training run {training_run_id} vanished mid-transaction"
                raise RuntimeError(msg)
            run.status = str(RunStatus.SUCCEEDED)
            run.finished_at = _utcnow()

            best_version_id = best_version.id

        return TrainingResult(
            run_uuid=run_uuid,
            training_run_id=training_run_id,
            candidates=candidates,
            best_model_name=best_candidate.name,
            best_model_version_id=best_version_id,
        )

    def _mark_failed(self, training_run_id: int, exc: Exception) -> None:
        """Record a training run as failed, in its own transaction."""
        logger.exception("Training run failed", extra={"training_run_id": training_run_id})
        with self._database.session_scope() as uow:
            run = uow.training_runs.get(training_run_id)
            if run is not None:
                run.status = str(RunStatus.FAILED)
                run.error = str(exc)
                run.finished_at = _utcnow()

    def _resolve_dataset_version(
        self, uow: UnitOfWork, dataset_version: str | None
    ) -> DatasetVersion:
        """Look up the dataset to train on, defaulting to the most recent build."""
        version = (
            uow.datasets.get_by_version(dataset_version)
            if dataset_version
            else uow.datasets.latest()
        )
        if version is None:
            target = dataset_version or "latest"
            msg = f"No dataset version found ({target}); run `f1cp dataset build` first"
            raise ValueError(msg)
        return version

    def _train_all_candidates(
        self,
        frame: pl.DataFrame,
        feature_names: list[str],
        run_uuid: str,
    ) -> list[CandidateResult]:
        """Fit, calibrate, score and select every enabled model-zoo candidate."""
        training = self._settings.training
        target_column = self._settings.features.target_column

        seasons = completed_seasons(frame)
        splitter = self._build_splitter(training.validation)
        folds = list(splitter.split(seasons))
        if not folds:
            msg = (
                f"Not enough completed seasons ({len(seasons)}) to form a single "
                "walk-forward fold under the current validation configuration"
            )
            raise ValueError(msg)

        zoo = build_zoo(self._settings)
        if not zoo:
            msg = "No model zoo entry is enabled in conf/models.yaml"
            raise ValueError(msg)

        return [
            self._train_one_candidate(
                spec, frame, feature_names, target_column, folds, seasons, run_uuid
            )
            for spec in zoo
        ]

    def _build_splitter(self, validation: ValidationConfig) -> Splitter:
        """Build the configured walk-forward splitter."""
        if validation.strategy == "rolling":
            return RollingWindowSplit(
                validation.rolling_window_seasons, validation.test_horizon_seasons
            )
        return SeasonForwardSplit(validation.min_train_seasons, validation.test_horizon_seasons)

    def _train_one_candidate(
        self,
        spec: ModelSpec,
        frame: pl.DataFrame,
        feature_names: list[str],
        target_column: str,
        folds: Sequence[Fold],
        seasons: list[int],
        run_uuid: str,
    ) -> CandidateResult:
        """Walk forward, score every fold, refit on everything, and log to MLflow."""
        training = self._settings.training

        fold_pooled: list[dict[str, float | None]] = []
        fold_phase: list[dict[str, dict[str, float | None]]] = []
        fold_frames: list[pl.DataFrame] = []
        for fold_index, fold in enumerate(folds):
            fit_seasons, calibration_seasons = _split_calibration_seasons(
                fold.train_seasons, training.validation.calibration_holdout_seasons
            )
            scored = self._fit_calibrate_score(
                spec,
                frame,
                feature_names,
                target_column,
                fit_seasons,
                calibration_seasons,
                fold.test_seasons,
            )
            fold_pooled.append(scored.pooled_metrics)
            fold_phase.append(scored.phase_metrics)
            fold_frames.append(scored.frame.with_columns(pl.lit(fold_index).alias("fold")))

        mean_pooled = _mean_metrics(fold_pooled)
        mean_phase = _mean_phase_metrics(fold_phase)
        oof_predictions_path = (
            self._settings.paths.processed / f"oof_{run_uuid}_{spec.name}.parquet"
        )
        pl.concat(fold_frames).write_parquet(oof_predictions_path)

        fit_seasons, calibration_seasons = _split_calibration_seasons(
            seasons, training.validation.calibration_holdout_seasons
        )
        final_pipeline, base_pipeline = self._fit_and_calibrate(
            spec, frame, feature_names, target_column, fit_seasons, calibration_seasons
        )

        mlflow_model_uri: str | None = None
        mlflow_base_model_uri: str | None = None
        with self._tracker.run(
            f"{run_uuid}-{spec.name}", tags={"run_uuid": run_uuid, "algorithm": spec.estimator_name}
        ) as mlflow_run_id:
            self._tracker.log_params(
                {
                    "model": spec.name,
                    "imbalance_strategy": training.imbalance.strategy,
                    "calibration_method": training.calibration.method,
                    **spec.params,
                }
            )
            self._tracker.log_metrics(
                {f"cv_{key}": value for key, value in mean_pooled.items() if value is not None}
            )
            for index, fold_metric in enumerate(fold_pooled):
                self._tracker.log_metrics(
                    {
                        f"fold{index}_{key}": value
                        for key, value in fold_metric.items()
                        if value is not None
                    }
                )
            mlflow_model_uri = self._tracker.log_model(final_pipeline, name=spec.name)
            if mlflow_model_uri:
                registered_name = f"{self._settings.mlflow.registered_model_name}-{spec.name}"
                self._tracker.register_model(mlflow_model_uri, registered_name)
            mlflow_base_model_uri = self._tracker.log_model(base_pipeline, name=f"{spec.name}-base")

        return CandidateResult(
            name=spec.name,
            estimator_name=spec.estimator_name,
            params=dict(spec.params),
            metrics=mean_pooled,
            phase_metrics=mean_phase,
            fold_metrics=fold_pooled,
            final_pipeline=final_pipeline,
            base_pipeline=base_pipeline,
            oof_predictions_path=str(oof_predictions_path),
            mlflow_run_id=mlflow_run_id,
            mlflow_model_uri=mlflow_model_uri,
            mlflow_base_model_uri=mlflow_base_model_uri,
        )

    def _fit_calibrate_score(
        self,
        spec: ModelSpec,
        frame: pl.DataFrame,
        feature_names: list[str],
        target_column: str,
        fit_seasons: list[int],
        calibration_seasons: list[int],
        test_seasons: list[int],
    ) -> _ScoredFold:
        """Fit on one fold's training seasons and score on its test season(s)."""
        training = self._settings.training
        model, _base_pipeline = self._fit_and_calibrate(
            spec, frame, feature_names, target_column, fit_seasons, calibration_seasons
        )

        test_frame = frame.filter(pl.col(_SEASON_COLUMN).is_in(test_seasons))
        x_test = _select_x(test_frame, feature_names)
        probabilities = model.predict_proba(x_test)[:, 1]
        scored_frame = test_frame.with_columns(pl.Series(_PROBABILITY_COLUMN, probabilities))

        if training.normalisation.enabled:
            normalizer = GroupProbabilityNormalizer(training.normalisation.group_columns)
            scored_frame = normalizer.normalize(scored_frame, _PROBABILITY_COLUMN)
        else:
            scored_frame = scored_frame.with_columns(
                pl.col(_PROBABILITY_COLUMN).alias(_NORMALIZED_COLUMN)
            )

        pooled = compute_metrics(
            scored_frame,
            calibrated_column=_PROBABILITY_COLUMN,
            normalized_column=_NORMALIZED_COLUMN,
            target_column=target_column,
            group_columns=training.normalisation.group_columns,
            calibration_bins=training.evaluation.calibration_bins,
        )
        pooled["final_round_top1_hit_rate"] = final_round_top1_hit_rate(
            scored_frame, probability_column=_NORMALIZED_COLUMN, target_column=target_column
        )

        phase_metrics: dict[str, dict[str, float | None]] = {}
        phase_expr = pl.lit(None, dtype=pl.String)
        for phase_name, (low, high) in training.evaluation.phase_bins.items():
            in_phase = (pl.col("season_progress") >= low) & (pl.col("season_progress") < high)
            phase_metrics[phase_name] = compute_metrics(
                scored_frame.filter(in_phase),
                calibrated_column=_PROBABILITY_COLUMN,
                normalized_column=_NORMALIZED_COLUMN,
                target_column=target_column,
                group_columns=training.normalisation.group_columns,
                calibration_bins=training.evaluation.calibration_bins,
            )
            phase_expr = pl.when(in_phase).then(pl.lit(phase_name)).otherwise(phase_expr)

        oof_frame = scored_frame.select(
            [
                *KEY_COLUMNS,
                phase_expr.alias("phase"),
                pl.col(target_column),
                pl.col(_PROBABILITY_COLUMN),
                pl.col(_NORMALIZED_COLUMN),
            ]
        )

        return _ScoredFold(pooled_metrics=pooled, phase_metrics=phase_metrics, frame=oof_frame)

    def _fit_and_calibrate(
        self,
        spec: ModelSpec,
        frame: pl.DataFrame,
        feature_names: list[str],
        target_column: str,
        fit_seasons: list[int],
        calibration_seasons: list[int],
    ) -> tuple[Pipeline, Pipeline]:
        """Fit a candidate on ``fit_seasons`` and calibrate it on ``calibration_seasons``.

        Returns:
            A ``(model, base_pipeline)`` pair: ``model`` is what gets scored and deployed
            (calibrated, if calibration is enabled); ``base_pipeline`` is the same fit
            *before* calibration, kept around for explainability. The two are the same
            object when calibration is disabled or skipped.
        """
        training = self._settings.training

        x_fit, y_fit = _select_xy(frame, feature_names, target_column, fit_seasons)
        positive = float(y_fit.sum())
        negative = float(len(y_fit)) - positive
        positive_weight = negative / positive if positive > 0 else None

        pipeline = spec.build_pipeline(
            imbalance_strategy=training.imbalance.strategy,
            random_state=self._settings.app.random_seed,
            positive_weight=positive_weight,
        )
        pipeline.fit(x_fit, y_fit)

        if training.calibration.enabled and calibration_seasons:
            x_calib, y_calib = _select_xy(frame, feature_names, target_column, calibration_seasons)
            calibrated = calibrate(pipeline, x_calib, y_calib, method=training.calibration.method)
            return calibrated, pipeline
        return pipeline, pipeline


def _select_x(frame: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Extract the model-input matrix as a dense, null-preserving-as-NaN array."""
    return frame.select(feature_names).cast(pl.Float64).to_numpy()


def _select_xy(
    frame: pl.DataFrame,
    feature_names: list[str],
    target_column: str,
    seasons: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the model-input matrix and label vector for a set of seasons."""
    subset = frame.filter(pl.col(_SEASON_COLUMN).is_in(seasons))
    return _select_x(subset, feature_names), subset[target_column].to_numpy()


def _split_calibration_seasons(seasons: list[int], holdout: int) -> tuple[list[int], list[int]]:
    """Split a fold's training seasons into a fit slice and a trailing calibration slice."""
    ordered = sorted(seasons)
    if holdout <= 0 or holdout >= len(ordered):
        return ordered, []
    return ordered[:-holdout], ordered[-holdout:]


def _mean_metrics(fold_metrics: Sequence[dict[str, float | None]]) -> dict[str, float | None]:
    """Average each metric across folds, ignoring folds where it could not be computed."""
    keys = {key for fold in fold_metrics for key in fold}
    result: dict[str, float | None] = {}
    for key in keys:
        values = [value for fold in fold_metrics if (value := fold.get(key)) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def _mean_phase_metrics(
    fold_phase_metrics: Sequence[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    """Average each phase's metrics across folds."""
    phases = {phase for fold in fold_phase_metrics for phase in fold}
    return {
        phase: _mean_metrics([fold[phase] for fold in fold_phase_metrics if phase in fold])
        for phase in phases
    }


def _evaluation_rows(model_version_id: int, candidate: CandidateResult) -> list[dict[str, object]]:
    """Flatten one candidate's pooled, per-phase and per-fold metrics into long-format rows."""
    rows: list[dict[str, object]] = []

    def _add(*, fold: int | None, phase: str | None, metrics: dict[str, float | None]) -> None:
        for metric_name, value in metrics.items():
            if value is None:
                continue
            rows.append(
                {
                    "model_version_id": model_version_id,
                    "split": "validation",
                    "fold": fold,
                    "season": None,
                    "phase": phase,
                    "metric_name": metric_name,
                    "metric_value": value,
                }
            )

    _add(fold=None, phase=None, metrics=candidate.metrics)
    for phase_name, phase_metrics in candidate.phase_metrics.items():
        _add(fold=None, phase=phase_name, metrics=phase_metrics)
    for fold_index, fold_metrics in enumerate(candidate.fold_metrics):
        _add(fold=fold_index, phase=None, metrics=fold_metrics)
    return rows


def _git_sha() -> str | None:
    """Best-effort current commit hash, or ``None`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _hash_training_config(settings: Settings) -> str:
    """Hash the configuration that determines a training run's content.

    Args:
        settings: Application settings.

    Returns:
        A hex-encoded SHA-256 digest, stable across runs given identical configuration.
    """
    payload = {
        "training": settings.training.model_dump(mode="json"),
        "model_zoo": {
            name: spec.model_dump(mode="json") for name, spec in settings.enabled_models().items()
        },
        "random_seed": settings.app.random_seed,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
