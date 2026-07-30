"""Tests for building model-zoo estimators and pipelines."""

from __future__ import annotations

import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from f1cp.models.base import ModelSpec, build_estimator

pytestmark = pytest.mark.unit

_IMPUTED_ESTIMATORS = ("logistic_regression", "random_forest", "mlp", "svm", "knn")
_NOT_IMPUTED_ESTIMATORS = ("xgboost", "lightgbm", "catboost")


@pytest.mark.parametrize("estimator_name", _IMPUTED_ESTIMATORS)
def test_build_pipeline_imputes_estimators_that_cannot_take_nan(estimator_name: str) -> None:
    spec = ModelSpec(
        name=estimator_name, estimator_name=estimator_name, params={}, scale_features=False
    )
    pipeline = spec.build_pipeline(imbalance_strategy="none", random_state=0)
    assert isinstance(pipeline.named_steps["impute"], SimpleImputer)


@pytest.mark.parametrize("estimator_name", _NOT_IMPUTED_ESTIMATORS)
def test_build_pipeline_does_not_impute_tree_based_estimators(estimator_name: str) -> None:
    spec = ModelSpec(
        name=estimator_name,
        estimator_name=estimator_name,
        params={"verbose": -1} if estimator_name == "lightgbm" else {},
        scale_features=False,
    )
    pipeline = spec.build_pipeline(imbalance_strategy="none", random_state=0)
    assert "impute" not in pipeline.named_steps


def test_build_pipeline_scales_features_when_configured() -> None:
    spec = ModelSpec(
        name="logistic_regression",
        estimator_name="logistic_regression",
        params={},
        scale_features=True,
    )
    pipeline = spec.build_pipeline(imbalance_strategy="none", random_state=0)
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)


def test_build_pipeline_skips_scaling_when_not_configured() -> None:
    spec = ModelSpec(
        name="random_forest", estimator_name="random_forest", params={}, scale_features=False
    )
    pipeline = spec.build_pipeline(imbalance_strategy="none", random_state=0)
    assert "scale" not in pipeline.named_steps


@pytest.mark.parametrize("estimator_name", ["logistic_regression", "random_forest", "lightgbm"])
def test_build_estimator_applies_class_weight_balanced(estimator_name: str) -> None:
    params = {"verbose": -1} if estimator_name == "lightgbm" else {}
    estimator = build_estimator(
        estimator_name, params, imbalance_strategy="balanced", random_state=0
    )
    assert estimator.class_weight == "balanced"


@pytest.mark.parametrize("estimator_name", ["logistic_regression", "random_forest", "lightgbm"])
def test_build_estimator_leaves_class_weight_unset_when_imbalance_disabled(
    estimator_name: str,
) -> None:
    params = {"verbose": -1} if estimator_name == "lightgbm" else {}
    estimator = build_estimator(estimator_name, params, imbalance_strategy="none", random_state=0)
    assert estimator.class_weight is None


def test_build_estimator_applies_xgboost_scale_pos_weight_when_balanced() -> None:
    estimator = build_estimator(
        "xgboost", {}, imbalance_strategy="balanced", random_state=0, positive_weight=4.0
    )
    assert estimator.scale_pos_weight == 4.0


def test_build_estimator_ignores_positive_weight_when_imbalance_disabled() -> None:
    estimator = build_estimator(
        "xgboost", {}, imbalance_strategy="none", random_state=0, positive_weight=4.0
    )
    assert estimator.scale_pos_weight == 1.0


def test_build_estimator_applies_catboost_balanced_class_weights() -> None:
    estimator = build_estimator(
        "catboost",
        {"verbose": 0, "allow_writing_files": False},
        imbalance_strategy="balanced",
        random_state=0,
    )
    assert estimator.get_params()["auto_class_weights"] == "Balanced"


def test_build_estimator_mlp_ignores_imbalance_strategy() -> None:
    balanced = build_estimator("mlp", {}, imbalance_strategy="balanced", random_state=0)
    none = build_estimator("mlp", {}, imbalance_strategy="none", random_state=0)
    assert not hasattr(balanced, "class_weight")
    assert not hasattr(none, "class_weight")


def test_build_estimator_svm_supports_probability_and_class_weight() -> None:
    estimator = build_estimator("svm", {}, imbalance_strategy="balanced", random_state=0)
    assert estimator.probability is True
    assert estimator.class_weight == "balanced"


def test_build_estimator_knn_ignores_imbalance_strategy() -> None:
    balanced = build_estimator("knn", {}, imbalance_strategy="balanced", random_state=0)
    none = build_estimator("knn", {}, imbalance_strategy="none", random_state=0)
    assert not hasattr(balanced, "class_weight")
    assert not hasattr(none, "class_weight")


def test_build_estimator_knn_has_no_random_state_parameter() -> None:
    """k-NN has no internal randomness, unlike every other candidate in the zoo."""
    estimator = build_estimator("knn", {}, imbalance_strategy="none", random_state=7)
    assert "random_state" not in estimator.get_params()


@pytest.mark.parametrize(
    "estimator_name",
    ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "mlp", "svm"],
)
def test_build_estimator_threads_random_state(estimator_name: str) -> None:
    params = {"verbose": -1} if estimator_name == "lightgbm" else {}
    if estimator_name == "catboost":
        params = {"verbose": 0, "allow_writing_files": False}
    estimator = build_estimator(estimator_name, params, imbalance_strategy="none", random_state=7)
    assert estimator.get_params()["random_state"] == 7


def test_build_estimator_rejects_an_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        build_estimator("linear_svm", {}, imbalance_strategy="none", random_state=0)
