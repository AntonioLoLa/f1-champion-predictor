"""Streamlit caching wrappers around the dashboard's slowest API calls.

``/predict``, ``/predict-season`` and ``/explain`` all rebuild the full feature panel from
the database on every call, by design: a served prediction must always reflect whatever has
actually been ingested, never a stale pre-computed snapshot (see
:mod:`f1cp.inference.predictor`). That guarantee is correct but not free -- measured against
the real bootstrapped database, a plain ``/predict`` costs several seconds and ``/explain``
(SHAP, slower still for algorithms with no fast native explainer -- k-NN, SVM, MLP) costs
tens of seconds.

Clicking between dashboard pages, or switching the model-version dropdown and switching it
straight back, re-issues the *same* request the user just made. A short-lived cache here
absorbs exactly that repeat-click cost without weakening the API's own freshness guarantee:
the API still recomputes from scratch every time *it* is actually called, and data in the
dashboard is never more than :data:`TTL_SECONDS` old.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

__all__ = ["TTL_SECONDS", "cached_explain", "cached_predict", "cached_predict_season"]

TTL_SECONDS = 60


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def cached_predict(
    _client: DashboardApiClient,
    *,
    season: int | None,
    round_number: int | None,
    model_version_id: int | None,
) -> dict[str, Any]:
    """Cached wrapper around :meth:`DashboardApiClient.predict`.

    Args:
        _client: The API client. Leading underscore so Streamlit does not try (and fail)
            to hash it; the cache key is the remaining, genuinely hashable arguments.
        season: Championship year, or ``None`` for the latest.
        round_number: Round within the season, or ``None`` for the latest.
        model_version_id: Model version to score with, or ``None`` for the active one.

    Returns:
        The same JSON `predict` would return.
    """
    return _client.predict(
        season=season, round_number=round_number, model_version_id=model_version_id
    )


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def cached_predict_season(
    _client: DashboardApiClient, season: int, *, model_version_id: int | None = None
) -> list[dict[str, Any]]:
    """Cached wrapper around :meth:`DashboardApiClient.predict_season`."""
    return _client.predict_season(season, model_version_id=model_version_id)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def cached_explain(
    _client: DashboardApiClient,
    season: int,
    round_number: int,
    *,
    model_version_id: int | None,
) -> dict[str, Any]:
    """Cached wrapper around :meth:`DashboardApiClient.explain`."""
    return _client.explain(season, round_number, model_version_id=model_version_id)
