"""Integration test for the training orchestrator.

Deliberately does not reuse ``tests/unit/features/synthetic.py``: that fixture is
pre-feature-pipeline source data spanning only two seasons, too little history to form a
single walk-forward fold. This builds a small, already-feature-engineered flat frame
instead -- exactly the shape :func:`~f1cp.dataset.builder.build_dataset` produces -- with
enough synthetic seasons to exercise expanding-origin CV end to end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from f1cp.db.models import DatasetVersion
from f1cp.db.session import Database
from f1cp.domain.enums import RunStatus
from f1cp.models.normalizer import GroupProbabilityNormalizer
from f1cp.training.trainer import Trainer

if TYPE_CHECKING:
    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration

_SEASONS = list(range(2000, 2009))  # nine completed seasons
_ROUNDS_PER_SEASON = 4
_DRIVERS_PER_SEASON = 4
_FEATURE_NAMES = ["skill", "experience"]


def _build_synthetic_frame() -> pl.DataFrame:
    """A small, deterministic flat feature matrix spanning several synthetic seasons.

    Each season's champion is ``driver_{season % _DRIVERS_PER_SEASON}``, and that driver's
    ``skill`` feature is set noticeably higher than their rivals' so the model zoo has real
    signal to learn, not just noise.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for season in _SEASONS:
        champion_index = season % _DRIVERS_PER_SEASON
        for round_number in range(1, _ROUNDS_PER_SEASON + 1):
            for driver_index in range(_DRIVERS_PER_SEASON):
                is_champion = driver_index == champion_index
                rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "driver_id": f"driver_{driver_index}",
                        "rounds_total": _ROUNDS_PER_SEASON,
                        "season_progress": round_number / _ROUNDS_PER_SEASON,
                        "current_points": round_number * (10.0 if is_champion else 5.0),
                        "is_mathematically_eliminated": False,
                        "champion": 1 if is_champion else 0,
                        "skill": (3.0 if is_champion else 0.0) + float(rng.normal(scale=0.5)),
                        "experience": float(driver_index) + float(rng.normal(scale=0.1)),
                    }
                )
    return pl.DataFrame(rows)


def _fast_settings(settings: Settings) -> Settings:
    """Shrink validation and the model zoo so the integration test runs quickly."""
    validation = settings.training.validation.model_copy(
        update={
            "strategy": "expanding",
            "min_train_seasons": 3,
            "test_horizon_seasons": 1,
            "calibration_holdout_seasons": 1,
        }
    )
    training = settings.training.model_copy(update={"validation": validation})
    fast_models = {"logistic_regression", "random_forest"}
    model_zoo = {
        name: spec.model_copy(update={"enabled": name in fast_models})
        for name, spec in settings.model_zoo.items()
    }
    return settings.model_copy(update={"training": training, "model_zoo": model_zoo})


def _register_dataset(database: Database, settings: Settings, frame: pl.DataFrame) -> None:
    path = settings.paths.processed / "dataset_test.parquet"
    frame.write_parquet(path)
    with database.session_scope() as uow:
        uow.datasets.add(
            DatasetVersion(
                version="test-version",
                path=str(path),
                n_rows=frame.height,
                n_features=len(_FEATURE_NAMES),
                min_season=min(_SEASONS),
                max_season=max(_SEASONS),
                feature_names=_FEATURE_NAMES,
                config_hash="test-hash",
            )
        )


def test_trainer_run_scores_selects_and_activates_a_model(settings: Settings) -> None:
    fast_settings = _fast_settings(settings)
    frame = _build_synthetic_frame()

    database = Database(fast_settings.database)
    database.create_all()
    _register_dataset(database, fast_settings, frame)

    trainer = Trainer(fast_settings, database)
    result = trainer.run()

    assert {candidate.name for candidate in result.candidates} == {
        "logistic_regression",
        "random_forest",
    }
    assert result.best_model_name in {"logistic_regression", "random_forest"}

    with database.session_scope() as uow:
        training_run = uow.training_runs.get(result.training_run_id)
        assert training_run is not None
        assert training_run.status == str(RunStatus.SUCCEEDED)
        assert training_run.finished_at is not None

        active = uow.models.get_active()
        assert active is not None
        assert active.id == result.best_model_version_id
        assert active.is_best is True

        model_versions = uow.models.list_versions()
        assert {mv.name for mv in model_versions} == {"logistic_regression", "random_forest"}

        evaluations = uow.evaluations.for_model(result.best_model_version_id)
        assert any(e.phase is None and e.fold is None for e in evaluations)  # pooled summary
        assert any(e.phase is not None for e in evaluations)  # phase-stratified
        assert any(e.fold is not None for e in evaluations)  # per-fold

    winner = next(c for c in result.candidates if c.name == result.best_model_name)
    final_round = frame.filter(
        (pl.col("season") == max(_SEASONS)) & (pl.col("round") == _ROUNDS_PER_SEASON)
    )
    x_test = final_round.select(_FEATURE_NAMES).cast(pl.Float64).to_numpy()
    probabilities = winner.final_pipeline.predict_proba(x_test)[:, 1]
    scored = final_round.with_columns(pl.Series("probability", probabilities))
    normalized = GroupProbabilityNormalizer(["season", "round"]).normalize(scored, "probability")
    assert normalized["probability_normalized"].sum() == pytest.approx(1.0)

    database.dispose()


def test_trainer_run_raises_without_enough_completed_seasons(settings: Settings) -> None:
    fast_settings = _fast_settings(settings)
    too_few_seasons = fast_settings.model_copy(
        update={
            "training": fast_settings.training.model_copy(
                update={
                    "validation": fast_settings.training.validation.model_copy(
                        update={"min_train_seasons": 100}
                    )
                }
            )
        }
    )
    frame = _build_synthetic_frame()

    database = Database(too_few_seasons.database)
    database.create_all()
    _register_dataset(database, too_few_seasons, frame)

    trainer = Trainer(too_few_seasons, database)
    with pytest.raises(ValueError, match="Not enough completed seasons"):
        trainer.run()

    with database.session_scope() as uow:
        runs = uow.training_runs.recent()
        assert len(runs) == 1
        assert runs[0].status == str(RunStatus.FAILED)
        assert runs[0].error is not None

    database.dispose()
