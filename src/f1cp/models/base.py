"""Model specifications: turning a config entry into a fittable sklearn pipeline.

Every algorithm in the zoo is reached through the same seam -- :class:`ModelSpec` -- so
the trainer never branches on "which algorithm is this". Two things vary per algorithm and
are handled here rather than by the trainer: whether it needs its inputs imputed and scaled
(linear/MLP models do; gradient-boosted trees natively handle missing values and are scale
invariant, and imputing them would erase a real signal -- "no circuit history yet" is itself
informative to a tree split), and how class imbalance is expressed in its constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

__all__ = ["ModelSpec", "build_estimator"]

# Algorithms whose sklearn implementation cannot take `NaN` directly. The gradient-boosted
# tree libraries (XGBoost, LightGBM, CatBoost) all handle missing values internally.
_NEEDS_IMPUTATION = frozenset({"logistic_regression", "random_forest", "mlp", "svm", "knn"})

# Algorithms with no native `class_weight`-style imbalance handling in their sklearn
# estimator; left unweighted, relying on calibration and per-group normalisation instead.
_NO_IMBALANCE_SUPPORT = frozenset({"mlp", "knn"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A named, buildable candidate from the model zoo.

    Attributes:
        name: Key from ``model_zoo`` in ``conf/models.yaml``, used for logging and the
            ``model_versions.algorithm`` column.
        estimator_name: Which algorithm to build (``logistic_regression``, ``xgboost``...).
        params: Hyperparameters passed straight to the estimator's constructor.
        scale_features: Whether to standardise inputs before fitting.
    """

    name: str
    estimator_name: str
    params: dict[str, Any]
    scale_features: bool

    def build_pipeline(
        self,
        *,
        imbalance_strategy: str,
        random_state: int,
        positive_weight: float | None = None,
    ) -> Pipeline:
        """Build an unfitted sklearn ``Pipeline`` for this candidate.

        Args:
            imbalance_strategy: ``"balanced"`` or ``"none"``, from
                ``training.imbalance.strategy``.
            random_state: Seed passed to the underlying estimator, from
                ``settings.app.random_seed``.
            positive_weight: Ratio of negative to positive examples in the training fold,
                used only by algorithms that take an explicit scale-pos-weight (XGBoost);
                ignored otherwise.

        Returns:
            A pipeline ready to ``fit(X, y)``.
        """
        estimator = build_estimator(
            self.estimator_name,
            self.params,
            imbalance_strategy=imbalance_strategy,
            random_state=random_state,
            positive_weight=positive_weight,
        )

        steps: list[tuple[str, Any]] = []
        if self.estimator_name in _NEEDS_IMPUTATION:
            steps.append(("impute", SimpleImputer(strategy="median")))
        if self.scale_features:
            steps.append(("scale", StandardScaler()))
        steps.append(("estimator", estimator))
        return Pipeline(steps)


def build_estimator(
    estimator_name: str,
    params: dict[str, Any],
    *,
    imbalance_strategy: str,
    random_state: int,
    positive_weight: float | None = None,
) -> ClassifierMixin:
    """Construct one estimator, wiring in class-imbalance handling per algorithm.

    Args:
        estimator_name: Which algorithm to build.
        params: Hyperparameters from configuration.
        imbalance_strategy: ``"balanced"`` or ``"none"``.
        random_state: Seed passed to the underlying estimator, from
            ``settings.app.random_seed``.
        positive_weight: Negative-to-positive class ratio for the training fold; only
            consumed by XGBoost's ``scale_pos_weight``.

    Returns:
        An unfitted, scikit-learn-compatible classifier.

    Raises:
        ValueError: If ``estimator_name`` is not one of the six supported algorithms.
    """
    balanced = imbalance_strategy == "balanced" and estimator_name not in _NO_IMBALANCE_SUPPORT

    if estimator_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            **params,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )

    if estimator_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **params,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )

    if estimator_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            scale_pos_weight=(positive_weight or 1.0) if balanced else 1.0,
            random_state=random_state,
            eval_metric="logloss",
        )

    if estimator_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )

    if estimator_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            auto_class_weights="Balanced" if balanced else None,
            random_state=random_state,
        )

    if estimator_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(**params, random_state=random_state)

    if estimator_name == "svm":
        from sklearn.svm import SVC

        # `probability=True` fits an internal 5-fold Platt-scaling calibrator so
        # `predict_proba` exists at all; it is what makes SVC noticeably slower to train
        # than the other margin/distance-based candidates here.
        return SVC(
            **params,
            probability=True,
            class_weight="balanced" if balanced else None,
            random_state=random_state,
        )

    if estimator_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        # No `random_state` parameter at all: unlike every other candidate here,
        # k-nearest-neighbours has no internal randomness to seed.
        return KNeighborsClassifier(**params)

    msg = f"Unknown estimator '{estimator_name}'"
    raise ValueError(msg)
