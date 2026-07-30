"""Tests for SHAP feature attribution."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from f1cp.config.settings import ShapConfig
from f1cp.explain.shap_explainer import explain

pytestmark = pytest.mark.unit

_FEATURE_NAMES = ["signal", "noise_a", "noise_b"]


def _synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """A feature matrix where only ``signal`` (column 0) predicts the label."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] + rng.normal(scale=0.3, size=300) > 0).astype(int)
    return x, y


@pytest.mark.parametrize(
    ("algorithm", "build_estimator"),
    [
        ("random_forest", lambda: RandomForestClassifier(n_estimators=50, random_state=0)),
        ("xgboost", lambda: XGBClassifier(n_estimators=50)),
        ("logistic_regression", lambda: LogisticRegression()),
    ],
)
def test_explain_ranks_the_predictive_feature_first(algorithm: str, build_estimator) -> None:  # type: ignore[no-untyped-def]
    x, y = _synthetic_data()
    pipeline = Pipeline([("impute", SimpleImputer()), ("estimator", build_estimator())])
    pipeline.fit(x, y)

    result = explain(
        pipeline,
        algorithm,
        x,
        _FEATURE_NAMES,
        ShapConfig(max_samples=100, background_samples=30),
        random_seed=0,
    )

    assert result.feature_names[0] == "signal"
    assert len(result.feature_names) == 3
    assert len(result.importances) == 3
    assert all(value >= 0 for value in result.importances)


def test_explain_handles_the_mlp_via_kernel_explainer() -> None:
    x, y = _synthetic_data()
    pipeline = Pipeline(
        [("impute", SimpleImputer()), ("estimator", MLPClassifier(max_iter=300, random_state=0))]
    )
    pipeline.fit(x, y)

    result = explain(
        pipeline,
        "mlp",
        x,
        _FEATURE_NAMES,
        ShapConfig(max_samples=50, background_samples=20),
        random_seed=0,
    )

    assert set(result.feature_names) == set(_FEATURE_NAMES)
    assert len(result.importances) == 3


def test_explain_caps_the_sample_at_the_available_row_count() -> None:
    x, y = _synthetic_data()
    pipeline = Pipeline([("estimator", RandomForestClassifier(n_estimators=10, random_state=0))])
    pipeline.fit(x, y)

    result = explain(
        pipeline,
        "random_forest",
        x[:20],
        _FEATURE_NAMES,
        ShapConfig(max_samples=2000, background_samples=100),
        random_seed=0,
    )
    assert len(result.feature_names) == 3
