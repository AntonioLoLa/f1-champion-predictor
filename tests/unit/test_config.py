"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from f1cp.config import PROJECT_ROOT, Settings
from f1cp.config.loader import deep_merge, expand_env_vars, load_yaml_files

pytestmark = pytest.mark.unit


class TestEnvExpansion:
    def test_uses_default_when_variable_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("F1CP_TEST_VAR", raising=False)
        assert expand_env_vars("${F1CP_TEST_VAR:-fallback}") == "fallback"

    def test_environment_wins_over_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("F1CP_TEST_VAR", "from-env")
        assert expand_env_vars("${F1CP_TEST_VAR:-fallback}") == "from-env"

    def test_expands_nested_structures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("F1CP_TEST_VAR", "x")
        payload = {"a": ["${F1CP_TEST_VAR}", {"b": "${F1CP_TEST_VAR}"}], "c": 3}
        assert expand_env_vars(payload) == {"a": ["x", {"b": "x"}], "c": 3}

    def test_missing_variable_without_default_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("F1CP_TEST_VAR", raising=False)
        with pytest.raises(KeyError, match="F1CP_TEST_VAR"):
            expand_env_vars("${F1CP_TEST_VAR}")


class TestDeepMerge:
    def test_merges_recursively_without_mutating_inputs(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 1}
        override = {"a": {"y": 99, "z": 3}}
        assert deep_merge(base, override) == {"a": {"x": 1, "y": 99, "z": 3}, "b": 1}
        assert base == {"a": {"x": 1, "y": 2}, "b": 1}

    def test_scalar_overrides_replace_mappings(self) -> None:
        assert deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}


class TestSettings:
    def test_loads_the_shipped_configuration(self, settings: Settings) -> None:
        assert settings.data.min_season == 1981
        assert settings.data.bootstrap_provider == "f1db"
        assert settings.data.incremental_provider == "jolpica"
        # svm and knn are disabled by default: both are significantly behind the leaders on
        # the selection metric and together were ~70% of a training run's wall clock.
        assert set(settings.enabled_models()) == {
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "mlp",
        }

    def test_never_exceeds_the_jolpica_page_limit(self, settings: Settings) -> None:
        """The server caps `limit` at 100; a larger value would silently truncate pages."""
        assert settings.data.providers.jolpica.page_limit <= 100

    def test_stays_within_the_documented_rate_limits(self, settings: Settings) -> None:
        jolpica = settings.data.providers.jolpica
        assert jolpica.requests_per_second <= 4.0
        assert jolpica.requests_per_hour <= 500

    def test_relative_sqlite_paths_are_anchored_to_the_project_root(self) -> None:
        """Otherwise the API and the trainer would use different database files."""
        from f1cp.config.settings import DatabaseConfig

        config = DatabaseConfig(url="sqlite:///data/f1cp.db")
        assert config.url == f"sqlite:///{(PROJECT_ROOT / 'data/f1cp.db').resolve()}"
        assert config.is_sqlite

    def test_absolute_and_memory_sqlite_urls_are_left_alone(self) -> None:
        from f1cp.config.settings import DatabaseConfig

        assert DatabaseConfig(url="sqlite:///:memory:").url == "sqlite:///:memory:"
        assert DatabaseConfig(url="postgresql+psycopg://u:p@h/db").is_sqlite is False

    def test_unknown_keys_are_rejected(self) -> None:
        from f1cp.config.settings import AppConfig

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            AppConfig(nmae="typo")  # type: ignore[call-arg]

    def test_max_season_defaults_to_the_current_year(self, settings: Settings) -> None:
        import datetime as dt

        assert settings.data.resolved_max_season == dt.date.today().year
        assert settings.data.min_season in settings.data.season_range

    def test_calibration_defaults_to_platt_without_the_yaml(self) -> None:
        """Isotonic on a three-season holdout reports 100% title odds."""
        from f1cp.config.settings import CalibrationConfig

        assert CalibrationConfig().method == "sigmoid"

    def test_missing_files_are_skipped(self, tmp_path: Path) -> None:
        assert load_yaml_files(tmp_path / "does-not-exist.yaml") == {}
