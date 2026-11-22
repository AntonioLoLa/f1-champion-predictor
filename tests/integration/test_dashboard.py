"""Integration test for the Streamlit dashboard, in-process against a real FastAPI app.

Reuses the same seeded database, dataset build and registered model as
``test_api.py``/``test_predictor.py``. Talks to the FastAPI app through Starlette's
``TestClient`` (an in-process, lifespan-aware ``httpx.Client``) instead of a real socket,
and drives each page through Streamlit's own ``AppTest.from_function`` -- the same
mechanism that runs when a real browser session connects, so a page that would raise on
first load fails this test instead of only being caught by manually clicking through the
app.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from f1cp.api.main import create_app
from f1cp.dashboard.api_client import DashboardApiClient
from f1cp.dashboard.components.labels import FEATURE_LABELS
from f1cp.db.session import Database
from tests.integration.test_dataset_builder import _seed
from tests.integration.test_predictor import _build_dataset, _fast_settings, _fit_and_register

if TYPE_CHECKING:
    from pathlib import Path

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration

_PAGES = (
    "home",
    "predictor",
    "archive",
    "driver_analysis",
    "model_performance",
    "feature_importance",
    "historical_predictions",
    "title_contention",
    "settings",
)


def _page_text(at: AppTest) -> str:
    """Every piece of text a page rendered, as one searchable string.

    The interface is built from :mod:`f1cp.dashboard.components.theme`'s own markup --
    headers, stat cards and leaderboards are HTML rendered through ``st.markdown`` rather
    than Streamlit's stock widgets -- so asserting that a page *says* something means
    searching its markdown, not reaching for a typed element like ``at.metric``.
    """
    return "\n".join(block.value for block in at.markdown)


def _run_page(module_name, client) -> None:
    """Render one dashboard page.

    Deliberately unannotated: ``AppTest.from_function`` re-execs only this function's own
    source text (not the rest of the module), so any name used here -- including in type
    annotations -- must be defined inside the function body itself.
    """
    import importlib

    module = importlib.import_module(f"f1cp.dashboard.pages.{module_name}")
    module.render(client)


def test_every_page_renders_without_error(settings: Settings, tmp_path: Path) -> None:
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    _fit_and_register(fast_settings, database, artifact)
    database.dispose()

    app = create_app(settings=fast_settings)
    with TestClient(app) as test_client:
        api_client = DashboardApiClient("http://testserver", timeout_seconds=30, client=test_client)

        failures = {}
        for page in _PAGES:
            at = AppTest.from_function(_run_page, args=(page, api_client), default_timeout=60)
            at.run()
            if at.exception:
                failures[page] = [str(exc) for exc in at.exception]

        assert not failures, failures


def test_predictor_page_shows_the_active_model(settings: Settings, tmp_path: Path) -> None:
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    _fit_and_register(fast_settings, database, artifact)
    database.dispose()

    app = create_app(settings=fast_settings)
    with TestClient(app) as test_client:
        api_client = DashboardApiClient("http://testserver", timeout_seconds=30, client=test_client)

        at = AppTest.from_function(_run_page, args=("predictor", api_client), default_timeout=60)
        at.run()

        assert not at.exception
        text = _page_text(at)
        # The page must say which model produced the numbers it is showing -- otherwise a
        # reader comparing two model versions cannot tell which one they are looking at.
        assert "Scored by Test model" in text
        assert at.dataframe
        # `_seed` truncates 2021 to its first two rounds, so that -- not 2020's final
        # round -- is the latest race with stored results the "use latest" default picks.
        assert "2021 · Round 2" in text


def test_home_page_warns_when_the_overview_is_in_sample(settings: Settings, tmp_path: Path) -> None:
    """The landing page must not present a replayed season as honest performance.

    ``_fit_and_register`` registers a model with no ``fit_seasons`` recorded, so
    ``is_out_of_sample`` reads ``False`` -- exactly the case this warning exists for.
    """
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    _fit_and_register(fast_settings, database, artifact)
    database.dispose()

    app = create_app(settings=fast_settings)
    with TestClient(app) as test_client:
        api_client = DashboardApiClient("http://testserver", timeout_seconds=30, client=test_client)

        at = AppTest.from_function(_run_page, args=("home", api_client), default_timeout=60)
        at.run()

        assert not at.exception
        assert at.warning
        assert "replay of a result it already knows" in at.warning[0].value


def test_title_contention_page_shows_who_is_still_alive(settings: Settings, tmp_path: Path) -> None:
    # No trained model needed: title contention is pure standings/calendar reasoning.
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    database.dispose()

    app = create_app(settings=fast_settings)
    with TestClient(app) as test_client:
        api_client = DashboardApiClient("http://testserver", timeout_seconds=30, client=test_client)

        at = AppTest.from_function(
            _run_page, args=("title_contention", api_client), default_timeout=60
        )
        at.run()
        at.selectbox(key="contention_season_only").set_value(2020)
        at.run()

        assert not at.exception
        # Of the synthetic championship's three drivers, exactly one can still take the
        # 2020 title after its final round -- the headline number the page exists to give.
        # It reads from `at.metric` because the stat cards are built on `st.metric`: a
        # native element keeps its label/value/detail hierarchy whether or not the app's
        # own stylesheet is applied, which raw markup cannot.
        assert at.metric
        assert "1 of 3" in [metric.value for metric in at.metric]


def test_every_feature_the_model_reads_has_a_readable_label(
    settings: Settings, tmp_path: Path
) -> None:
    """A feature added without a label would show its column name on the attribution page."""
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    database.dispose()

    unlabelled = sorted(set(artifact.feature_names) - set(FEATURE_LABELS))
    assert not unlabelled, f"Add these to dashboard/components/labels.py: {unlabelled}"


def test_pickers_offer_seasons_with_results_and_races_by_name(
    settings: Settings, tmp_path: Path
) -> None:
    fast_settings = _fast_settings(settings, tmp_path)
    database = Database(fast_settings.database)
    database.create_all()
    _seed(database)
    artifact = _build_dataset(database, fast_settings)
    _fit_and_register(fast_settings, database, artifact)
    database.dispose()
    st.cache_data.clear()

    app = create_app(settings=fast_settings)
    with TestClient(app) as test_client:
        api_client = DashboardApiClient("http://testserver", timeout_seconds=30, client=test_client)
        seasons = [
            row["season"] for row in api_client.seasons()["seasons"] if row["rounds_completed"]
        ]

        at = AppTest.from_function(_run_page, args=("predictor", api_client), default_timeout=60)
        at.run()
        at.toggle(key="predictor_use_latest").set_value(False)
        at.run()
        at.selectbox(key="predictor_season_only").set_value(2020)
        at.run()

        assert not at.exception
        assert at.selectbox(key="predictor_season_only").options == [str(s) for s in seasons]
        grand_prix = at.selectbox(key="predictor_round_2020").options
        # One option per round run, labelled with the race's name rather than a bare number.
        assert len(grand_prix) == 6
        assert all(option.startswith(f"R{n} · ") for n, option in enumerate(grand_prix, 1))
        # Switching season lands on that season's latest race, not the previous selection.
        assert "2020 · Round 6" in _page_text(at)


class _CannedClient:
    """Just enough of :class:`DashboardApiClient` for the overview, with fixed answers.

    The overview's headline depends on states the synthetic championship never reaches at
    its latest round -- a title already decided, a field tied at 0% -- so it is driven
    here from responses shaped exactly like the API's.
    """

    def __init__(self, drivers: list[dict[str, Any]], *, alive: int, rounds_left: int) -> None:
        self._drivers = drivers
        self._alive = alive
        self._rounds_left = rounds_left

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "database": True, "read_only": True}

    def predict(self, **_: Any) -> dict[str, Any]:
        return {
            "season": 2026,
            "round": 20,
            "model_version_id": 49,
            "model_name": "logistic_regression",
            "is_out_of_sample": True,
            "drivers": self._drivers,
        }

    def title_contention(self, season: int, *, round_number: int | None = None) -> dict[str, Any]:
        return {
            "season": season,
            "round": round_number,
            "rounds_remaining": self._rounds_left,
            "contenders": [
                {"still_alive": index < self._alive} for index in range(len(self._drivers))
            ],
        }

    def standings(self, season: int, *, round_number: int | None = None) -> dict[str, Any]:
        return {
            "season": season,
            "round": round_number,
            "standings": [
                {
                    "driver_id": driver["driver_id"],
                    "driver_name": driver["driver_name"],
                    "constructor_id": driver["constructor_id"],
                    "constructor_name": driver["constructor_name"],
                    "points": driver["points"],
                    "position": driver["standing_position"],
                    "wins": 0,
                }
                for driver in self._drivers
            ],
        }


def _canned_driver(
    rank: int, name: str, team: tuple[str, str], points: float, position: int, probability: float
) -> dict[str, Any]:
    return {
        "driver_id": name.lower().replace(" ", "-"),
        "driver_name": name,
        "constructor_id": team[0],
        "constructor_name": team[1],
        "points": points,
        "standing_position": position,
        "probability": probability,
        "probability_calibrated": probability,
        "probability_raw": probability,
        "rank": rank,
    }


_MERCEDES = ("mercedes", "Mercedes")
_RED_BULL = ("red-bull", "Red Bull")
_WILLIAMS = ("williams", "Williams")


def _stats(at: AppTest) -> dict[str, tuple[str, str]]:
    """Each headline card as ``label -> (value, detail)``."""
    return {metric.label: (metric.value, str(metric.delta)) for metric in at.metric}


def test_a_decided_title_is_announced_rather_than_shown_as_100_percent() -> None:
    # 2026 after the Brazilian GP, as the API ranks it: the field tied at 0% follows the
    # standings, so Verstappen (P2) -- not Albon (P15, first alphabetically) -- is second.
    client = _CannedClient(
        [
            _canned_driver(1, "Kimi Antonelli", _MERCEDES, 390.0, 1, 1.0),
            _canned_driver(2, "Max Verstappen", _RED_BULL, 279.0, 2, 0.0),
            _canned_driver(3, "George Russell", _MERCEDES, 262.0, 3, 0.0),
            _canned_driver(4, "Alexander Albon", _WILLIAMS, 40.0, 15, 0.0),
        ],
        alive=1,
        rounds_left=3,
    )
    st.cache_data.clear()

    at = AppTest.from_function(_run_page, args=("home", client), default_timeout=60)
    at.run()

    assert not at.exception
    stats = _stats(at)
    assert stats["Champion"][0] == "Kimi Antonelli"
    assert "Mercedes" in stats["Champion"][1]
    assert stats["Winning margin"] == ("111 pts", "over Max Verstappen, P2 · 279 pts")
    assert stats["Rounds to spare"][0] == "3"
    assert "Title chance" not in stats
    assert "Closest challenger" not in stats
    # Teams by name in the odds panel; never the slug.
    text = _page_text(at)
    assert "Red Bull" in text
    assert "red-bull" not in text


def test_an_open_title_compares_odds_without_a_misleading_difference() -> None:
    client = _CannedClient(
        [
            _canned_driver(1, "Kimi Antonelli", _MERCEDES, 300.0, 1, 0.62),
            _canned_driver(2, "Max Verstappen", _RED_BULL, 280.0, 2, 0.30),
            _canned_driver(3, "George Russell", _MERCEDES, 262.0, 3, 0.08),
        ],
        alive=3,
        rounds_left=6,
    )
    st.cache_data.clear()

    at = AppTest.from_function(_run_page, args=("home", client), default_timeout=60)
    at.run()

    assert not at.exception
    stats = _stats(at)
    assert stats["Title chance"] == ("62.0%", "vs 30.0% for Max Verstappen")
    assert stats["Closest challenger"] == ("Max Verstappen", "30.0% · P2 · 280 pts")
    assert stats["Can still win"][0] == "3"
    assert "Champion" not in stats
