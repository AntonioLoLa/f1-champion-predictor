"""Resolve the model zoo from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1cp.models.base import ModelSpec

if TYPE_CHECKING:
    from f1cp.config.settings import Settings

__all__ = ["build_zoo"]


def build_zoo(settings: Settings) -> list[ModelSpec]:
    """Build every enabled candidate from ``conf/models.yaml``.

    Args:
        settings: Application settings.

    Returns:
        One :class:`~f1cp.models.base.ModelSpec` per enabled entry in ``model_zoo``, in
        declaration order.
    """
    return [
        ModelSpec(
            name=name,
            estimator_name=spec.estimator,
            params=dict(spec.params),
            scale_features=spec.scale_features,
        )
        for name, spec in settings.enabled_models().items()
    ]
