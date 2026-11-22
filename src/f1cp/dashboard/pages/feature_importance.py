"""Which inputs drove the model's predictions for one round, via SHAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_explain, cached_predict
from f1cp.dashboard.components.charts import feature_importance_chart
from f1cp.dashboard.components.labels import feature_label
from f1cp.dashboard.components.layout import (
    control_panel,
    model_version_selector,
    season_round_inputs,
    show_api_error,
)
from f1cp.dashboard.components.theme import Stat, card, page_header, section, stat_grid

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_TITLE = "What drives the odds"
_SUBTITLE = (
    "Which facts about each driver -- points, form, team, the calendar still to come -- "
    "moved the model's title odds most for a chosen round."
)


def render(client: DashboardApiClient) -> None:
    """Render the feature attribution page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Explanation")
        season, round_number = season_round_inputs(client, "explain")
        model_version_id = model_version_selector(client, "explain")
        top_n = st.slider("Inputs shown", min_value=5, max_value=30, value=15)

    if season is None or round_number is None:
        with st.spinner("Resolving the latest round..."):
            try:
                latest = cached_predict(
                    client, season=None, round_number=None, model_version_id=None
                )
            except ApiError as error:
                page_header(_TITLE, _SUBTITLE)
                show_api_error(error)
                return
        season, round_number = latest["season"], latest["round"]

    with st.spinner("Working out what drove the odds -- this can take a while the first time..."):
        try:
            result = cached_explain(client, season, round_number, model_version_id=model_version_id)
        except ApiError as error:
            page_header(_TITLE, _SUBTITLE)
            show_api_error(error)
            return

    names = [feature_label(name) for name in result["feature_names"]]
    importances = result["importances"]
    total = sum(importances) or 1.0
    top_share = sum(importances[:5]) / total

    page_header(
        _TITLE,
        _SUBTITLE,
        meta=[(f"{result['season']} · Round {result['round']}", "accent")],
    )

    stat_grid(
        [
            Stat(
                label="Most influential",
                value=names[0] if names else "-",
                detail="moved the odds most, on average",
                tone="accent",
            ),
            Stat(
                label="Top five share",
                value=f"{top_share:.0%}",
                detail="of the total influence, for the five strongest",
            ),
            Stat(
                label="Inputs considered",
                value=str(len(names)),
                detail="facts the model reads about every driver",
            ),
        ]
    )

    section(
        "What moved the prediction",
        "How far each input pushed a driver's odds up or down, averaged over the field. "
        "Longer bars mattered more; the direction is not shown. Measured with SHAP, a "
        "standard way of splitting a prediction between its inputs.",
    )
    with card():
        st.plotly_chart(
            feature_importance_chart(names, importances, top_n=top_n),
            width="stretch",
            config={"displayModeBar": False},
        )
