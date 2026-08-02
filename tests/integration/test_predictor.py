"""Integration test for live inference against a real (SQLite) database and MLflow store.

Reuses the synthetic championship (:mod:`tests.unit.features.synthetic`) and its seeding
helper from :mod:`tests.integration.test_dataset_builder`: this is exactly the shape of
data :class:`~f1cp.inference.predictor.PredictionService` reads at serving time, unlike
``test_trainer.py``'s already-flat synthetic frame. A trivial real ``LogisticRegression``
pipeline is fit on the built dataset and logged to a throwaway MLflow store, so the service
exercises the genuine MLflow load path rather than a stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from f1cp.dataset.builder import DatasetArtifact, build_dataset
from f1cp.db.models import DataIngestion, DatasetVersion, ModelVersion, TrainingRun
from f1cp.db.session import Database
from f1cp.domain.enums import RunStatus
from f1cp.inference.predictor import PredictionService
from f1cp.training.tracking import ExperimentTracker
from tests.integration.test_dataset_builder import _seed
from tests.unit.features.synthetic import ROUNDS, build_synthetic_sources

if TYPE_CHECKING:
    from pathlib import Path

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


def _fast_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Shrink the season filter and point MLflow at a throwaway store."""
    return settings.model_copy(
        update={
            "features": settings.features.model_copy(
                update={
                    "filters": settings.features.filters.model_copy(
                        update={"min_rounds_in_season": 1}
                    )
                }
            ),
            "mlflow": settings.mlflow.model_copy(
                update={
                    "enabled": True,
                    "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                }
            ),
        }
    )


def _build_dataset(database: Database, settings: Settings) -> DatasetArtifact:
    """Build the dataset and register it, exactly as ``f1cp dataset build`` does.

    Registration matters beyond bookkeeping: :meth:`~f1cp.training.trainer.Trainer.run`
    resolves its input dataset via ``uow.datasets.latest()``, so a training run started
    through the API has nothing to train on without this.
    """
    with database.session_scope() as uow:
        artifact = build_dataset(uow, settings)
        uow.datasets.add(
            DatasetVersion(
                version=artifact.version,
                path=artifact.path,
                n_rows=artifact.n_rows,
                n_features=artifact.n_features,
                min_season=artifact.min_season,
                max_season=artifact.max_season,
                feature_names=artifact.feature_names,
                config_hash=artifact.config_hash,
            )
        )
        return artifact


def _fit_and_register(settings: Settings, database: Database, artifact: DatasetArtifact) -> int:
    """Fit a real model on the built dataset, log it to MLflow, and activate it.

    Returns:
        The primary key of the now-active :class:`~f1cp.db.models.ModelVersion`.
    """
    frame = artifact.frame
    x = frame.select(artifact.feature_names).cast(pl.Float64).to_numpy()
    y = frame[settings.features.target_column].to_numpy()

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("estimator", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(x, y)

    tracker = ExperimentTracker(settings.mlflow)
    with tracker.run("test-predictor"):
        model_uri = tracker.log_model(pipeline, name="test-model")
    assert model_uri is not None

    with database.session_scope() as uow:
        training_run = TrainingRun(run_uuid="test-run", status=str(RunStatus.SUCCEEDED))
        uow.training_runs.add(training_run)
        model_version = ModelVersion(
            training_run_id=training_run.id,
            name="test-model",
            algorithm="logistic_regression",
            mlflow_model_uri=model_uri,
            mlflow_base_model_uri=model_uri,
            artifact_path="models/test-model",
            is_active=True,
            feature_names=artifact.feature_names,
        )
        uow.models.add(model_version)
        return model_version.id


def _prepare(settings: Settings, tmp_path: Path) -> tuple[Settings, Database, int]:
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    model_version_id = _fit_and_register(fast_settings, database, artifact)
    return fast_settings, database, model_version_id


def test_predict_round_returns_ranked_normalised_probabilities(
    settings: Settings, tmp_path: Path
) -> None:
    fast_settings, database, model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    with database.session_scope() as uow:
        result = service.predict_round(uow, season=2020, round_number=4)

    assert result.season == 2020
    assert result.round == 4
    assert result.model_version_id == model_version_id
    assert {driver.driver_id for driver in result.drivers} == {"d1", "d2", "d3"}
    assert sum(driver.probability for driver in result.drivers) == pytest.approx(1.0)

    ranked_probabilities = [driver.probability for driver in result.drivers]
    assert ranked_probabilities == sorted(ranked_probabilities, reverse=True)
    assert [driver.rank for driver in result.drivers] == [1, 2, 3]

    with database.session_scope() as uow:
        persisted = uow.predictions.for_round(2020, 4, model_version_id=model_version_id)
    assert len(persisted) == 3
    assert sum(row.probability_normalised or 0.0 for row in persisted) == pytest.approx(1.0)

    database.dispose()


def test_predict_round_defaults_to_the_latest_result(settings: Settings, tmp_path: Path) -> None:
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    with database.session_scope() as uow:
        result = service.predict_round(uow)

    # `_seed` (borrowed from test_dataset_builder) truncates 2021 to its first two rounds
    # to simulate a title fight still in progress, so that -- not 2020's final round -- is
    # the latest race with stored results.
    assert (result.season, result.round) == (2021, 2)
    database.dispose()


def test_predict_season_scores_every_round(settings: Settings, tmp_path: Path) -> None:
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    with database.session_scope() as uow:
        results = service.predict_season(uow, 2020)

    assert [result.round for result in results] == list(ROUNDS)
    for result in results:
        assert sum(driver.probability for driver in result.drivers) == pytest.approx(1.0)

    database.dispose()


def test_explain_round_returns_feature_importances(settings: Settings, tmp_path: Path) -> None:
    fast_settings, database, model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    with database.session_scope() as uow:
        result = service.explain_round(uow, season=2020, round_number=4)

    assert result.model_version_id == model_version_id
    assert result.feature_names
    assert len(result.feature_names) == len(result.importances)
    assert all(importance >= 0 for importance in result.importances)

    database.dispose()


def test_predict_round_raises_for_unknown_model_version(settings: Settings, tmp_path: Path) -> None:
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    with (
        database.session_scope() as uow,
        pytest.raises(ValueError, match="No model version found"),
    ):
        service.predict_round(uow, season=2020, round_number=4, model_version_id=999)

    database.dispose()


def test_title_contention_identifies_the_sole_survivor_at_season_end(
    settings: Settings, tmp_path: Path
) -> None:
    # No model registered at all: title contention is pure standings/calendar reasoning,
    # with zero dependency on a trained model.
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    service = PredictionService(fast_settings)

    final_round = ROUNDS[-1]
    with database.session_scope() as uow:
        champions = uow.standings.champion_by_season()
        result = service.title_contention(uow, season=2020, round_number=final_round)

    assert result.season == 2020
    assert result.round == final_round
    assert len(result.contenders) == 3

    still_alive = [c.driver_id for c in result.contenders if c.still_alive]
    assert still_alive == [champions[2020]]

    database.dispose()


def _count_panel_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch in a counter around the feature pipeline; returns a one-element tally."""
    from f1cp.inference import predictor as predictor_module

    calls = [0]
    original = predictor_module.run_pipeline

    def counting(*args: object, **kwargs: object) -> pl.DataFrame:
        calls[0] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(predictor_module, "run_pipeline", counting)
    return calls


def test_repeat_predictions_reuse_the_engineered_panel(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching model version must cost a rescore, not a full feature rebuild.

    The panel is the expensive part of a prediction and depends only on the source data,
    so a second request against unchanged data -- whatever model it names, whichever
    endpoint it came from -- must not rebuild it.
    """
    fast_settings, database, model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)
    calls = _count_panel_builds(monkeypatch)

    with database.session_scope() as uow:
        first = service.predict_round(uow, season=2020, round_number=4)
        service.predict_round(uow, season=2020, round_number=3)
        service.predict_round(uow, season=2020, round_number=4, model_version_id=model_version_id)
        service.explain_round(uow, season=2020, round_number=4)
        repeated = service.predict_round(uow, season=2020, round_number=4)

    assert calls[0] == 1
    assert [driver.probability for driver in repeated.drivers] == [
        driver.probability for driver in first.drivers
    ]

    database.dispose()


def test_new_results_invalidate_the_cached_panel(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freshly ingested rows must be reflected immediately, never served from the cache."""
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)
    calls = _count_panel_builds(monkeypatch)

    with database.session_scope() as uow:
        before = service.predict_round(uow)
    assert (before.season, before.round) == (2021, 2)

    # Run the third round of 2021, which `_seed` deliberately left unrun.
    sources = build_synthetic_sources()
    with database.session_scope() as uow:
        race_ids = uow.races.id_map()
        new_rows = []
        for row in sources.race_results.filter(
            (pl.col("season") == 2021) & (pl.col("round") == 3)
        ).to_dicts():
            season, round_number = row.pop("season"), row.pop("round")
            new_rows.append({**row, "race_id": race_ids[(season, round_number)]})
        uow.results.upsert_race_results(new_rows)

    with database.session_scope() as uow:
        after = service.predict_round(uow, season=2021, round_number=3)

    assert calls[0] == 2
    assert after.round == 3

    database.dispose()


def test_an_ingestion_run_invalidates_the_cached_panel(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row counts alone are not enough: a refresh rewrites rows in place.

    ``f1cp data update --refresh`` re-fetches seasons that are already stored and upserts
    over them -- a corrected classification changes values while leaving every table
    exactly the same size. The ingestion audit trail is what makes that visible.
    """
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)
    calls = _count_panel_builds(monkeypatch)

    with database.session_scope() as uow:
        service.predict_round(uow, season=2020, round_number=4)

    with database.session_scope() as uow:
        uow.ingestions.add(
            DataIngestion(provider="jolpica", mode="refresh", status=str(RunStatus.SUCCEEDED))
        )

    with database.session_scope() as uow:
        service.predict_round(uow, season=2020, round_number=4)

    assert calls[0] == 2

    database.dispose()


def test_repeat_explanations_are_memoised(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHAP is the single most expensive call the dashboard makes, and it is deterministic.

    Revisiting the Feature Importance page must not recompute an explanation the service
    has already produced for the same data, model and round -- but new data must.
    """
    from f1cp.inference import predictor as predictor_module

    fast_settings, database, model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings)

    calls = [0]
    original = predictor_module.explain_shap

    def counting(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(predictor_module, "explain_shap", counting)

    with database.session_scope() as uow:
        first = service.explain_round(uow, season=2020, round_number=4)
        again = service.explain_round(uow, season=2020, round_number=4)
        service.explain_round(uow, season=2020, round_number=4, model_version_id=model_version_id)

    assert calls[0] == 1
    assert again.feature_names == first.feature_names
    assert again.importances == first.importances

    with database.session_scope() as uow:
        uow.ingestions.add(
            DataIngestion(provider="jolpica", mode="refresh", status=str(RunStatus.SUCCEEDED))
        )
    with database.session_scope() as uow:
        service.explain_round(uow, season=2020, round_number=4)

    assert calls[0] == 2

    database.dispose()


def test_panel_caching_can_be_switched_off(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast_settings, database, _model_version_id = _prepare(settings, tmp_path)
    service = PredictionService(fast_settings, panel_cache_size=0)
    calls = _count_panel_builds(monkeypatch)

    with database.session_scope() as uow:
        service.predict_round(uow, season=2020, round_number=4)
        service.predict_round(uow, season=2020, round_number=4)

    assert calls[0] == 2

    database.dispose()


def test_title_contention_defaults_to_the_latest_result(settings: Settings, tmp_path: Path) -> None:
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    service = PredictionService(fast_settings)

    with database.session_scope() as uow:
        result = service.title_contention(uow)

    # `_seed` truncates 2021 to its first two rounds, so that is the latest race with
    # stored results, exactly as in the equivalent `predict_round` default test.
    assert (result.season, result.round) == (2021, 2)

    database.dispose()
