"""Permutation importance for the deployed model.

Unlike SHAP, permutation importance operates directly on the full pipeline (imputer/
scaler and all) against the *original* feature matrix: shuffling a raw column and
re-scoring the whole pipeline end to end is exactly what the technique measures, so the
input here should never be pre-transformed the way :mod:`f1cp.explain.shap_explainer`
needs to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.inspection import permutation_importance

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from f1cp.config.settings import PermutationImportanceConfig

__all__ = ["PermutationResult", "explain"]


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """Mean and standard deviation of importance per feature, most important first.

    Attributes:
        feature_names: Feature names, ordered by descending mean importance.
        importances_mean: Mean drop in negative log loss when each feature is shuffled.
        importances_std: Standard deviation across repeats, same order.
    """

    feature_names: list[str]
    importances_mean: list[float]
    importances_std: list[float]


def explain(
    pipeline: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: PermutationImportanceConfig,
    *,
    random_seed: int,
) -> PermutationResult:
    """Score the drop in negative log loss when each feature is independently shuffled.

    Args:
        pipeline: The fitted, pre-calibration pipeline to explain.
        x: The full, un-transformed feature matrix.
        y: The champion label for each row of ``x``.
        feature_names: Column names of ``x``, in order.
        config: Permutation importance settings (repeat count).
        random_seed: Seed for the shuffling.

    Returns:
        Feature importances, most important first.
    """
    result = permutation_importance(
        pipeline,
        x,
        y,
        n_repeats=config.n_repeats,
        random_state=random_seed,
        scoring="neg_log_loss",
    )
    order = np.argsort(result.importances_mean)[::-1]
    return PermutationResult(
        feature_names=[feature_names[i] for i in order],
        importances_mean=[float(result.importances_mean[i]) for i in order],
        importances_std=[float(result.importances_std[i]) for i in order],
    )
