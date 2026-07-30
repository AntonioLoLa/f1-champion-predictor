"""Tests for the MLflow experiment tracking wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sklearn.linear_model import LogisticRegression

from f1cp.config.settings import MLflowConfig
from f1cp.training.tracking import ExperimentTracker

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_disabled_tracker_is_a_no_op() -> None:
    tracker = ExperimentTracker(MLflowConfig(enabled=False))

    with tracker.run("a-run") as run_id:
        assert run_id is None
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2.0})

    assert tracker.log_model(LogisticRegression(), "x") is None
    assert tracker.register_model("models:/x/1", "x") is None


def test_enabled_tracker_logs_a_full_run(tmp_path: Path) -> None:
    config = MLflowConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="test-experiment",
        registered_model_name="test-model",
    )
    tracker = ExperimentTracker(config)

    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])

    with tracker.run("candidate-run", tags={"fold": "0"}) as run_id:
        assert run_id is not None
        tracker.log_params({"C": 1.0})
        tracker.log_metrics({"group_log_loss": 0.42})
        model_uri = tracker.log_model(model, "candidate")

    assert model_uri is not None
    assert model_uri.startswith("runs:/") or model_uri.startswith("models:/")

    version = tracker.register_model(model_uri, "test-model")
    assert version is not None


def test_artifact_location_anchors_a_newly_created_experiment(tmp_path: Path) -> None:
    import mlflow

    config = MLflowConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="anchored-experiment",
    )
    artifacts_dir = tmp_path / "artifacts"
    ExperimentTracker(config, artifact_location=f"file://{artifacts_dir}")

    experiment = mlflow.get_experiment_by_name("anchored-experiment")
    assert experiment is not None
    assert experiment.artifact_location == f"file://{artifacts_dir}"


def test_artifact_location_does_not_move_an_existing_experiment(tmp_path: Path) -> None:
    import mlflow

    config = MLflowConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="pre-existing-experiment",
    )
    # First tracker creates the experiment with no explicit location (MLflow's own default).
    ExperimentTracker(config)
    original_location = mlflow.get_experiment_by_name("pre-existing-experiment").artifact_location

    # A second tracker passing a location must not retroactively move it.
    ExperimentTracker(config, artifact_location=f"file://{tmp_path / 'elsewhere'}")
    assert (
        mlflow.get_experiment_by_name("pre-existing-experiment").artifact_location
        == original_location
    )
