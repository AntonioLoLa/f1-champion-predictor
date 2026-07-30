"""Integration test for evaluation report generation.

Extends ``test_trainer.py``'s synthetic fixture: trains a model, then builds a report
against the resulting active model version, entirely from what training persisted (the
out-of-fold predictions parquet), without re-fitting anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from f1cp.config.settings import MLflowConfig
from f1cp.db.session import Database
from f1cp.evaluation.report import build_report
from f1cp.training.trainer import Trainer
from tests.integration.test_trainer import (
    _build_synthetic_frame,
    _fast_settings,
    _register_dataset,
)

if TYPE_CHECKING:
    from f1cp.config.settings import Settings
    from f1cp.training.trainer import TrainingResult

pytestmark = pytest.mark.integration


def _trained_database(settings: Settings) -> tuple[Database, TrainingResult]:
    """Train the fast synthetic model zoo and return the database and the training result."""
    fast_settings = _fast_settings(settings)
    frame = _build_synthetic_frame()

    database = Database(fast_settings.database)
    database.create_all()
    _register_dataset(database, fast_settings, frame)

    trainer = Trainer(fast_settings, database)
    result = trainer.run()
    return database, result


def test_build_report_scores_the_active_model_and_writes_a_report(settings: Settings) -> None:
    database, result = _trained_database(settings)
    best_model_version_id = result.best_model_version_id

    with database.session_scope() as uow:
        before = len(uow.evaluations.for_model(best_model_version_id))
        report = build_report(uow, settings, model_version_id=best_model_version_id)
        after = len(uow.evaluations.for_model(best_model_version_id))

    report_dir = Path(report.report_dir)
    assert report.model_version_id == best_model_version_id
    assert report_dir.is_dir()
    assert (report_dir / "index.html").is_file()
    assert (report_dir / "metrics.json").is_file()
    assert (report_dir / "calibration_curve.json").is_file()

    # New metrics were added on top of training's, never re-inserting what's already there.
    assert after > before
    assert "accuracy" in report.pooled_metrics
    assert "group_log_loss" in report.pooled_metrics  # read back from training, not recomputed
    assert set(report.phase_metrics) <= {"early", "mid", "late"}
    assert report.season_metrics  # per-season is a new granularity training never persists

    database.dispose()


def test_build_report_pooled_metrics_match_what_training_reported(settings: Settings) -> None:
    """Regression test: pooled metrics read back from training must be the *pooled* row,
    not an arbitrary phase or season row sharing the same metric name."""
    database, result = _trained_database(settings)
    best_model_version_id = result.best_model_version_id
    winner = next(c for c in result.candidates if c.name == result.best_model_name)

    with database.session_scope() as uow:
        report = build_report(uow, settings, model_version_id=best_model_version_id)

    for metric_name in (
        "group_log_loss",
        "brier_score",
        "expected_calibration_error",
        "top1_hit_rate",
    ):
        assert report.pooled_metrics[metric_name] == pytest.approx(winner.metrics[metric_name])
        for phase_name in report.phase_metrics:
            assert report.phase_metrics[phase_name][metric_name] == pytest.approx(
                winner.phase_metrics[phase_name][metric_name]
            )

    database.dispose()


def test_build_report_is_idempotent_across_repeated_runs(settings: Settings) -> None:
    database, result = _trained_database(settings)
    best_model_version_id = result.best_model_version_id

    with database.session_scope() as uow:
        build_report(uow, settings, model_version_id=best_model_version_id)
        first_count = len(uow.evaluations.for_model(best_model_version_id))

    with database.session_scope() as uow:
        build_report(uow, settings, model_version_id=best_model_version_id)
        second_count = len(uow.evaluations.for_model(best_model_version_id))

    assert second_count == first_count
    database.dispose()


def test_build_report_defaults_to_the_active_model(settings: Settings) -> None:
    database, result = _trained_database(settings)

    with database.session_scope() as uow:
        report = build_report(uow, settings)

    assert report.model_version_id == result.best_model_version_id
    database.dispose()


def test_build_report_raises_for_an_unknown_model_version(settings: Settings) -> None:
    database, _result = _trained_database(settings)

    with database.session_scope() as uow, pytest.raises(ValueError, match="No model version found"):
        build_report(uow, settings, model_version_id=999_999)

    database.dispose()


def test_build_report_runs_explainability_end_to_end(settings: Settings, tmp_path: Path) -> None:
    mlflow_settings = settings.model_copy(
        update={
            "mlflow": MLflowConfig(
                enabled=True,
                tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                experiment_name="test-experiment",
                registered_model_name="test-model",
            )
        }
    )

    database, result = _trained_database(mlflow_settings)

    # Simulate `evaluate` running as its own process, with nothing else in it having
    # pointed MLflow at the configured store first (e.g. by constructing a `Trainer`).
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'somewhere-else.db'}")

    with database.session_scope() as uow:
        report = build_report(uow, mlflow_settings, model_version_id=result.best_model_version_id)
    database.dispose()

    report_dir = Path(report.report_dir)
    assert (report_dir / "shap_importance.json").is_file()
    assert (report_dir / "permutation_importance.json").is_file()
