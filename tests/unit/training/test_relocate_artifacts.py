"""Tests for repointing a shipped sqlite MLflow store at a new checkout."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from f1cp.training.tracking import relocate_artifact_paths

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_LAPTOP = "/home/dev/f1cp"
_RUNNER = "/home/runner/work/f1cp/f1cp"


def _store(path: Path, artifact_uris: list[str]) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE experiments (artifact_location TEXT)")
    con.execute("CREATE TABLE runs (artifact_uri TEXT)")
    con.execute("CREATE TABLE logged_models (artifact_location TEXT)")
    con.execute("CREATE TABLE model_versions (storage_location TEXT)")
    con.executemany("INSERT INTO runs VALUES (?)", [(uri,) for uri in artifact_uris])
    con.commit()
    con.close()


def _runs(path: Path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [row[0] for row in con.execute("SELECT artifact_uri FROM runs ORDER BY rowid")]
    finally:
        con.close()


def test_paths_from_every_training_machine_are_repointed(tmp_path: Path) -> None:
    """A store holds models from the laptop *and* the CI runner once CI retrains."""
    db = tmp_path / "mlflow.db"
    _store(
        db,
        [
            f"{_LAPTOP}/mlruns/1/abc/artifacts",
            f"file://{_RUNNER}/mlruns/1/def/artifacts",
        ],
    )
    relocate_artifact_paths(db, tmp_path / "checkout")

    assert _runs(db) == [
        f"{tmp_path}/checkout/mlruns/1/abc/artifacts",
        f"{tmp_path}/checkout/mlruns/1/def/artifacts",
    ]


def test_relocation_is_idempotent_and_ignores_other_values(tmp_path: Path) -> None:
    db = tmp_path / "mlflow.db"
    _store(db, [f"{_LAPTOP}/mlruns/1/abc/artifacts", "s3://bucket/elsewhere"])
    relocate_artifact_paths(db, tmp_path)
    once = _runs(db)
    relocate_artifact_paths(db, tmp_path)

    assert _runs(db) == once
    assert once[1] == "s3://bucket/elsewhere"
