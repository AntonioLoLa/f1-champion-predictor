"""Tests for resolving the model zoo from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from f1cp.models.base import ModelSpec
from f1cp.models.zoo import build_zoo

if TYPE_CHECKING:
    from f1cp.config.settings import Settings

pytestmark = pytest.mark.unit


def test_build_zoo_returns_every_enabled_entry_from_conf_models_yaml(settings: Settings) -> None:
    zoo = build_zoo(settings)
    names = {spec.name for spec in zoo}
    assert names == set(settings.enabled_models())
    assert all(isinstance(spec, ModelSpec) for spec in zoo)


def test_build_zoo_skips_disabled_entries(settings: Settings) -> None:
    disabled = settings.model_copy(
        update={
            "model_zoo": {
                **settings.model_zoo,
                "mlp": settings.model_zoo["mlp"].model_copy(update={"enabled": False}),
            }
        }
    )
    zoo = build_zoo(disabled)
    assert "mlp" not in {spec.name for spec in zoo}


def test_build_zoo_carries_estimator_and_params_through(settings: Settings) -> None:
    zoo = build_zoo(settings)
    by_name = {spec.name: spec for spec in zoo}
    xgboost = by_name["xgboost"]
    assert xgboost.estimator_name == "xgboost"
    assert xgboost.params["max_depth"] == 4
    assert xgboost.scale_features is False
