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

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from f1cp.api.main import create_app
from f1cp.dashboard.api_client import DashboardApiClient
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
        assert "Scored by test-model" in text
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
        at.number_input[0].set_value(2020)
        at.run()

        assert not at.exception
        # Of the synthetic championship's three drivers, exactly one can still take the
        # 2020 title after its final round -- the headline number the page exists to give.
        # It reads from `at.metric` because the stat cards are built on `st.metric`: a
        # native element keeps its label/value/detail hierarchy whether or not the app's
        # own stylesheet is applied, which raw markup cannot.
        assert at.metric
        assert "1 of 3" in [metric.value for metric in at.metric]
