"""Tests for permutation importance."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from f1cp.config.settings import PermutationImportanceConfig
from f1cp.explain.permutation import explain

pytestmark = pytest.mark.unit

_FEATURE_NAMES = ["signal", "noise_a", "noise_b"]


def test_explain_ranks_the_predictive_feature_first() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = (x[:, 0] + rng.normal(scale=0.3, size=300) > 0).astype(int)

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer()),
            ("estimator", RandomForestClassifier(n_estimators=100, random_state=0)),
        ]
    )
    pipeline.fit(x, y)

    result = explain(
        pipeline,
        x,
        y,
        _FEATURE_NAMES,
        PermutationImportanceConfig(n_repeats=10),
        random_seed=0,
    )

    assert result.feature_names[0] == "signal"
    assert len(result.importances_mean) == 3
    assert len(result.importances_std) == 3
    # Shuffling the genuinely predictive feature should hurt more than shuffling noise.
    signal_index = result.feature_names.index("signal")
    noise_index = result.feature_names.index("noise_a")
    assert result.importances_mean[signal_index] > result.importances_mean[noise_index]
