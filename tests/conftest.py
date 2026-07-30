"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from f1cp.config import CONF_DIR, load_settings

if TYPE_CHECKING:
    from f1cp.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding recorded provider payloads."""
    return FIXTURES_DIR


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Project settings redirected at a throwaway data directory and SQLite database."""
    monkeypatch.setenv("F1CP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    loaded = load_settings(CONF_DIR)
    loaded.paths.ensure_exist()
    return loaded
