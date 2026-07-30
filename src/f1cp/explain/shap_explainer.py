"""SHAP feature attribution for the deployed model.

Explains :attr:`~f1cp.training.trainer.CandidateResult.base_pipeline` -- the fitted
pipeline *before* calibration (see that module's docstring for why) -- dispatching to the
fast, algorithm-native explainer PLAN.md calls for: ``TreeExplainer`` for the gradient-
boosted/ensemble models, ``LinearExplainer`` for logistic regression, and a sampled
``KernelExplainer`` for the MLP (the one architecture with no fast native path). Every
pipeline may have an imputer/scaler ahead of the estimator, so the pipeline is split into
its preprocessing prefix and final estimator -- the explainer sees exactly what the
estimator sees, never the raw un-preprocessed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import shap

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from f1cp.config.settings import ShapConfig

    class _Explanation(Protocol):
        values: np.ndarray


__all__ = ["ShapResult", "explain"]

_TREE_ALGORITHMS = frozenset({"random_forest", "xgboost", "lightgbm", "catboost"})
_LINEAR_ALGORITHMS = frozenset({"logistic_regression"})


@dataclass(frozen=True, slots=True)
class ShapResult:
    """Mean absolute SHAP value per feature, sorted most important first.

    Attributes:
        feature_names: Feature names, ordered by descending importance.
        importances: Mean absolute SHAP value for each feature, same order.
    """

    feature_names: list[str]
    importances: list[float]


def _split_pipeline(pipeline: Pipeline) -> tuple[Pipeline | None, Any]:
    """Split a fitted pipeline into its preprocessing prefix and final estimator."""
    estimator = pipeline.steps[-1][1]
    preprocessing = pipeline[:-1] if len(pipeline.steps) > 1 else None
    return preprocessing, estimator


def _transform(preprocessing: Pipeline | None, x: np.ndarray) -> np.ndarray:
    return x if preprocessing is None else preprocessing.transform(x)


def _positive_class_values(explanation: _Explanation) -> np.ndarray:
    """Normalise SHAP output to ``(n_samples, n_features)`` for the positive class.

    Some tree explainers (notably scikit-learn's ``RandomForestClassifier``) return one
    slice per class; others (XGBoost, LightGBM, CatBoost, linear models) already return
    the positive-class contribution directly.
    """
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def explain(
    pipeline: Pipeline,
    algorithm: str,
    x: np.ndarray,
    feature_names: list[str],
    config: ShapConfig,
    *,
    random_seed: int,
) -> ShapResult:
    """Compute mean |SHAP value| per feature for a sample of rows.

    Args:
        pipeline: The fitted, pre-calibration pipeline to explain.
        algorithm: The model zoo algorithm identifier, used to pick the explainer.
        x: The full feature matrix (rows the model would actually see), pre-imputation.
        feature_names: Column names of ``x``, in order.
        config: SHAP settings (sample sizes).
        random_seed: Seed for the row sample and, for the MLP, the background sample.

    Returns:
        Feature importances, most important first.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = min(config.max_samples, x.shape[0])
    sample_idx = rng.choice(x.shape[0], size=n_samples, replace=False)

    preprocessing, estimator = _split_pipeline(pipeline)
    x_transformed = _transform(preprocessing, x[sample_idx])

    if algorithm in _TREE_ALGORITHMS:
        explainer = shap.TreeExplainer(estimator)
    elif algorithm in _LINEAR_ALGORITHMS:
        explainer = shap.LinearExplainer(estimator, x_transformed)
    else:
        background_size = min(config.background_samples, x_transformed.shape[0])
        background = shap.sample(x_transformed, background_size)
        explainer = shap.KernelExplainer(lambda arr: estimator.predict_proba(arr)[:, 1], background)

    values = _positive_class_values(explainer(x_transformed))
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    return ShapResult(
        feature_names=[feature_names[i] for i in order],
        importances=[float(mean_abs[i]) for i in order],
    )
