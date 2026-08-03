"""Integration test for the FastAPI service against a real (SQLite) database.

Builds the actual ASGI app via :func:`f1cp.api.main.create_app` -- never the module-level
``app``, which loads settings from the ambient environment -- against the same synthetic
championship and registered model used by ``test_predictor.py``, and drives it through
``TestClient`` exactly as a real client would over HTTP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from f1cp.api.main import create_app
from f1cp.db.session import Database
from f1cp.domain.enums import RunStatus
from tests.integration.test_dataset_builder import _seed
from tests.integration.test_predictor import _build_dataset, _fast_settings, _fit_and_register
from tests.unit.features.synthetic import ROUNDS

if TYPE_CHECKING:
    from pathlib import Path

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


def _client(settings: Settings, tmp_path: Path) -> tuple[TestClient, int]:
    """Seed a database, register a real model, and return a ready ``TestClient``."""
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    model_version_id = _fit_and_register(fast_settings, database, artifact)
    database.dispose()

    app = create_app(settings=fast_settings)
    return TestClient(app), model_version_id


def test_health_reports_the_active_model(settings: Settings, tmp_path: Path) -> None:
    client, _model_version_id = _client(settings, tmp_path)
    with client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["active_model"] == "test-model"


def test_model_endpoints_list_and_activate(settings: Settings, tmp_path: Path) -> None:
    client, model_version_id = _client(settings, tmp_path)
    with client:
        active = client.get("/model")
        assert active.status_code == 200
        assert active.json()["id"] == model_version_id

        versions = client.get("/model/versions")
        assert versions.status_code == 200
        assert [v["id"] for v in versions.json()] == [model_version_id]

        activated = client.post(f"/model/{model_version_id}/activate")
        assert activated.status_code == 200
        assert activated.json()["is_active"] is True

        missing = client.post("/model/999999/activate")
        assert missing.status_code == 404


def test_predict_and_predict_season_endpoints(settings: Settings, tmp_path: Path) -> None:
    client, model_version_id = _client(settings, tmp_path)
    with client:
        predicted = client.post("/predict", json={"season": 2020, "round": 4})
        assert predicted.status_code == 200
        body = predicted.json()
        assert body["model_version_id"] == model_version_id
        assert len(body["drivers"]) == 3
        assert sum(d["probability"] for d in body["drivers"]) == pytest.approx(1.0)

        season = client.get("/predict-season/2020")
        assert season.status_code == 200
        assert [round_["round"] for round_ in season.json()] == list(ROUNDS)

        not_found = client.post("/predict", json={"season": 1900, "round": 1})
        assert not_found.status_code == 404


def test_explain_endpoint(settings: Settings, tmp_path: Path) -> None:
    client, model_version_id = _client(settings, tmp_path)
    with client:
        response = client.get("/explain/2020/4")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version_id"] == model_version_id
    assert len(body["feature_names"]) == len(body["importances"])


def test_standings_endpoint(settings: Settings, tmp_path: Path) -> None:
    client, _model_version_id = _client(settings, tmp_path)
    with client:
        response = client.get("/seasons/2020/standings")

    assert response.status_code == 200
    body = response.json()
    assert body["round"] == ROUNDS[-1]
    assert {row["driver_id"] for row in body["standings"]} == {"d1", "d2", "d3"}
    positions = [row["position"] for row in body["standings"]]
    assert positions == sorted(positions)


def test_title_contention_endpoint(settings: Settings, tmp_path: Path) -> None:
    client, _model_version_id = _client(settings, tmp_path)
    with client:
        response = client.get("/seasons/2020/title-contention")

    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2020
    assert body["round"] == ROUNDS[-1]
    assert len(body["contenders"]) == 3
    # At the season's final round, exactly one driver is still mathematically alive.
    still_alive = [c for c in body["contenders"] if c["still_alive"]]
    assert len(still_alive) == 1
    assert still_alive[0]["eliminated_by"] is None
    for contender in body["contenders"]:
        if not contender["still_alive"]:
            assert contender["eliminated_by"] is not None
            assert contender["eliminated_by_name"] is not None


def test_train_endpoint_records_a_failed_run_without_enough_seasons(
    settings: Settings, tmp_path: Path
) -> None:
    # The synthetic championship has only three seasons; TestClient runs FastAPI's
    # background tasks to completion before `post` returns, so the run has already
    # failed (not enough seasons for a walk-forward fold) by the time it is polled.
    client, _model_version_id = _client(settings, tmp_path)
    with client:
        scheduled = client.post("/train", json={})
        assert scheduled.status_code == 202
        run_uuid = scheduled.json()["run_uuid"]

        status = client.get(f"/train/{run_uuid}")

    assert status.status_code == 200
    body = status.json()
    assert body["status"] == str(RunStatus.FAILED)
    assert body["error"] is not None


def test_read_only_mode_blocks_mutating_endpoints_but_not_reads(
    settings: Settings, tmp_path: Path
) -> None:
    read_only_settings = settings.model_copy(
        update={"api": settings.api.model_copy(update={"read_only": True})}
    )
    client, model_version_id = _client(read_only_settings, tmp_path)
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["read_only"] is True

        # Reads stay open -- read-only blocks writes, not the whole API.
        active = client.get("/model")
        assert active.status_code == 200

        activated = client.post(f"/model/{model_version_id}/activate")
        assert activated.status_code == 403

        trained = client.post("/train", json={})
        assert trained.status_code == 403

        updated = client.post("/update-data", json={})
        assert updated.status_code == 403
