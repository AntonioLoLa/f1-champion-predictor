"""Tests for probability calibration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from f1cp.models.calibration import calibrate

pytestmark = pytest.mark.unit


def _fitted_pipeline() -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)
    model = LogisticRegression().fit(x, y)
    return model, x, y


def test_calibrate_does_not_change_the_base_estimators_coefficients() -> None:
    model, x, y = _fitted_pipeline()
    original_coef = model.coef_.copy()

    calibrated = calibrate(model, x, y, method="isotonic")

    assert np.array_equal(model.coef_, original_coef)
    assert calibrated is not model


def test_calibrate_produces_valid_probabilities() -> None:
    model, x, y = _fitted_pipeline()
    calibrated = calibrate(model, x, y, method="sigmoid")

    probabilities = calibrated.predict_proba(x)
    assert probabilities.shape == (len(y), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrate_accepts_both_configured_methods(method: str) -> None:
    model, x, y = _fitted_pipeline()
    calibrated = calibrate(model, x, y, method=method)
    assert calibrated.predict_proba(x).shape == (len(y), 2)
