"""Integration tests for the historical results archive.

These endpoints serve the ingested record rather than anything the model produced, so the
whole suite runs against a seeded database with no training involved -- which is itself
the property worth asserting: the archive works before a model exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from f1cp.api.main import create_app
from f1cp.db.session import Database
from tests.integration.test_dataset_builder import _seed
from tests.integration.test_predictor import _fast_settings
from tests.unit.features.synthetic import ROUNDS

if TYPE_CHECKING:
    from pathlib import Path

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def client(settings: Settings, tmp_path: Path):
    """An API client over a seeded database, with no model trained."""
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)

    with database.session_scope() as uow:
        race_id = uow.races.id_map()[(2020, 1)]
        uow.results.upsert_pit_stops(
            [
                {"race_id": race_id, "driver_id": "d1", "stop": 1, "lap": 12, "duration_ms": 23145},
                {"race_id": race_id, "driver_id": "d2", "stop": 1, "lap": 14, "duration_ms": 21980},
            ]
        )
    database.dispose()

    with TestClient(create_app(settings=fast_settings)) as test_client:
        yield test_client


def test_seasons_lists_completed_and_in_progress_seasons(client) -> None:
    body = client.get("/seasons").json()

    seasons = {row["season"]: row for row in body["seasons"]}
    assert set(seasons) == {2019, 2020, 2021}
    # Newest first, so the season a reader most likely wants is the default.
    assert [row["season"] for row in body["seasons"]] == [2021, 2020, 2019]

    # `_seed` runs 2019 and 2020 in full but truncates 2021 to its first two rounds of its
    # calendar, so only 2019 and 2020 have a champion.
    assert seasons[2019]["champion_name"]
    assert seasons[2019]["rounds_completed"] == len(ROUNDS)
    assert seasons[2020]["champion_name"]
    assert seasons[2020]["rounds_completed"] == len(ROUNDS)
    assert seasons[2021]["champion_name"] is None
    assert seasons[2021]["rounds"] == len(ROUNDS)
    assert seasons[2021]["rounds_completed"] == 2


def test_season_races_returns_the_full_calendar_including_unrun_rounds(client) -> None:
    """A calendar is published before it is run; omitting future rounds would hide it."""
    races = {race["round"]: race for race in client.get("/seasons/2021/races").json()["races"]}

    # `_seed` truncates 2021 to its first two rounds of a published, longer calendar.
    assert sorted(races) == list(ROUNDS)
    assert races[3]["winner_id"] is None
    assert races[4]["winner_id"] is None
    # Round 2 is the first round of 2021 the fixture gives an outright winner: its round 1
    # has a retirement and no classified P1 at all, which the archive reports faithfully
    # as "no winner" rather than promoting whoever finished best.
    assert races[2]["winner_id"] == "d2"
    assert races[2]["winner_name"]
    assert races[1]["winner_id"] is None


def test_race_results_are_classified_winner_first(client) -> None:
    results = client.get("/seasons/2020/races/1/results").json()

    assert results
    assert results[0]["position"] == 1
    positions = [row["position_order"] for row in results]
    assert positions == sorted(positions)


def test_race_results_404_for_a_round_that_has_not_been_run(client) -> None:
    response = client.get("/seasons/2021/races/4/results")

    assert response.status_code == 404


def test_pit_stops_lists_stops_for_a_race(client) -> None:
    stops = client.get("/seasons/2020/races/1/pit-stops").json()

    assert [row["driver_id"] for row in stops] == ["d1", "d2"]
    assert stops[0]["stop"] == 1
    assert stops[0]["lap"] == 12
    assert stops[0]["driver_name"]


def test_pit_stops_is_empty_but_not_404_for_a_race_with_no_recorded_stops(client) -> None:
    response = client.get("/seasons/2020/races/2/pit-stops")

    assert response.status_code == 200
    assert response.json() == []


def test_pit_stops_404_for_a_round_that_has_not_been_run(client) -> None:
    response = client.get("/seasons/2021/races/4/pit-stops")

    assert response.status_code == 404


def test_constructor_standings_default_to_the_latest_completed_round(client) -> None:
    latest = client.get("/seasons/2020/constructor-standings").json()
    explicit = client.get(
        "/seasons/2020/constructor-standings", params={"round": ROUNDS[-1]}
    ).json()

    assert latest == explicit
    assert latest
    positions = [row["position"] for row in latest]
    assert positions == sorted(positions)


def test_archive_works_without_any_trained_model(client) -> None:
    """The point of the archive: it is the record, not a model output.

    `/model` has nothing to serve on this fixture, and none of the archive endpoints care.
    """
    assert client.get("/model").status_code == 404

    for path in (
        "/seasons",
        "/seasons/2020/races",
        "/seasons/2020/races/1/results",
        "/seasons/2020/races/1/pit-stops",
        "/seasons/2020/constructor-standings",
    ):
        assert client.get(path).status_code == 200, path
