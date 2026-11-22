"""Streamlit dashboard entrypoint.

Talks to the FastAPI service exclusively over HTTP -- see :mod:`f1cp.dashboard.api_client`
-- never to the database or MLflow directly, so it can be deployed, scaled and redeployed
independently of the API and the training pipeline. ``scripts/run_dashboard.py`` is the
thin wrapper that launches this module via ``streamlit run``.

Pages are grouped into four sections that follow how the app is actually read: what the
model says about the title right now, the analysis behind it, the model that produced it,
and the pipeline that feeds it.
"""

from __future__ import annotations

import streamlit as st

from f1cp.config import load_settings
from f1cp.dashboard.api_client import DashboardApiClient
from f1cp.dashboard.pages import (
    archive,
    driver_analysis,
    feature_importance,
    historical_predictions,
    home,
    model_performance,
    predictor,
    title_contention,
)
from f1cp.dashboard.pages import settings as settings_page


@st.cache_resource
def _client() -> DashboardApiClient:
    """Build (once per process) the HTTP client every page shares."""
    dashboard_settings = load_settings().dashboard
    return DashboardApiClient(
        dashboard_settings.api_base_url,
        timeout_seconds=dashboard_settings.request_timeout_seconds,
    )


def main() -> None:
    """Configure the page and run whichever section the user navigates to."""
    settings = load_settings()
    st.set_page_config(
        page_title=settings.dashboard.page_title,
        page_icon=":material/trophy:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    client = _client()

    # Every callback is a lambda, so `st.Page` must be told the URL path explicitly --
    # left to infer it, it uses the callable's `__name__`, and every lambda shares the
    # name "<lambda>", which collides across pages. Icons are Material Symbols, not emoji,
    # and every one of them names its page's subject rather than decorating it.
    navigation = st.navigation(
        {
            "Championship": [
                st.Page(
                    lambda: home.render(client),
                    title="Overview",
                    icon=":material/speed:",
                    url_path="home",
                    default=True,
                ),
                st.Page(
                    lambda: predictor.render(client),
                    title="Title odds",
                    icon=":material/trophy:",
                    url_path="predictor",
                ),
                st.Page(
                    lambda: title_contention.render(client),
                    title="Elimination",
                    icon=":material/functions:",
                    url_path="title-contention",
                ),
            ],
            "Archive": [
                st.Page(
                    lambda: archive.render(client),
                    title="Results archive",
                    icon=":material/inventory_2:",
                    url_path="archive",
                ),
            ],
            "Analysis": [
                st.Page(
                    lambda: driver_analysis.render(client),
                    title="Driver trajectory",
                    icon=":material/timeline:",
                    url_path="driver-analysis",
                ),
                st.Page(
                    lambda: historical_predictions.render(client),
                    title="Season replay",
                    icon=":material/history:",
                    url_path="historical-predictions",
                ),
            ],
            "Model": [
                st.Page(
                    lambda: model_performance.render(client),
                    title="Leaderboard",
                    icon=":material/leaderboard:",
                    url_path="model-performance",
                ),
                st.Page(
                    lambda: feature_importance.render(client),
                    title="What drives the odds",
                    icon=":material/insights:",
                    url_path="feature-importance",
                ),
            ],
            "System": [
                st.Page(
                    lambda: settings_page.render(client),
                    title="Pipeline",
                    icon=":material/settings:",
                    url_path="settings",
                ),
            ],
        }
    )
    navigation.run()


main()
