"""Probability calibration on a held-out slice of training seasons.

A model selected for discrimination is not automatically well calibrated: a driver a model
ranks correctly can still get an implausible raw probability. Calibration is fit on seasons
the model has never been trained on -- ``FrozenEstimator`` stops ``CalibratedClassifierCV``
from refitting the base estimator, so this step only ever adjusts the probability mapping,
never the model itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

if TYPE_CHECKING:
    import numpy as np
    from sklearn.pipeline import Pipeline

__all__ = ["calibrate"]


def calibrate(
    fitted_pipeline: Pipeline,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    *,
    method: str,
) -> CalibratedClassifierCV:
    """Fit a probability calibrator on top of an already-fitted pipeline.

    Args:
        fitted_pipeline: A pipeline already fit on the training seasons.
        x_calibration: Feature matrix for the held-out calibration seasons.
        y_calibration: Labels for the held-out calibration seasons.
        method: ``"isotonic"`` or ``"sigmoid"``.

    Returns:
        A fitted :class:`~sklearn.calibration.CalibratedClassifierCV` wrapping the frozen
        pipeline; ``.predict_proba`` returns calibrated probabilities.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method)
    calibrated.fit(x_calibration, y_calibration)
    return calibrated
