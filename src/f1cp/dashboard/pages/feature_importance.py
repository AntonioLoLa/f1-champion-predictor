"""Which inputs drove the model's predictions for one round, via SHAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_explain, cached_predict
from f1cp.dashboard.components.charts import feature_importance_chart
from f1cp.dashboard.components.layout import (
    control_panel,
    model_version_selector,
    season_round_inputs,
    show_api_error,
)
from f1cp.dashboard.components.theme import Stat, card, page_header, section, stat_grid

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_SUBTITLE = (
    "What the model actually weighed when it scored a round, measured with SHAP against "
    "the drivers who contested it."
)


def render(client: DashboardApiClient) -> None:
    """Render the feature attribution page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Explanation")
        season, round_number = season_round_inputs("explain")
        model_version_id = model_version_selector(client, "explain")
        top_n = st.slider("Features shown", min_value=5, max_value=30, value=15)

    if season is None or round_number is None:
        with st.spinner("Resolving the latest round..."):
            try:
                latest = cached_predict(
                    client, season=None, round_number=None, model_version_id=None
                )
            except ApiError as error:
                page_header("Feature attribution", _SUBTITLE)
                show_api_error(error)
                return
        season, round_number = latest["season"], latest["round"]

    with st.spinner("Computing SHAP attributions -- this can take a while on first load..."):
        try:
            result = cached_explain(client, season, round_number, model_version_id=model_version_id)
        except ApiError as error:
            page_header("Feature attribution", _SUBTITLE)
            show_api_error(error)
            return

    names = result["feature_names"]
    importances = result["importances"]
    total = sum(importances) or 1.0
    top_share = sum(importances[:5]) / total

    page_header(
        "Feature attribution",
        _SUBTITLE,
        meta=[(f"{result['season']} · Round {result['round']}", "accent")],
    )

    stat_grid(
        [
            Stat(
                label="Strongest input",
                value=names[0] if names else "-",
                detail="largest mean absolute SHAP value",
                tone="accent",
            ),
            Stat(
                label="Top five share",
                value=f"{top_share:.0%}",
                detail="of all attribution mass",
            ),
            Stat(
                label="Features in the model",
                value=str(len(names)),
                detail="engineered for this round",
            ),
        ]
    )

    section(
        "What moved the prediction",
        "Mean absolute SHAP value per feature: how much each input shifted the model's "
        "output, regardless of direction.",
    )
    with card():
        st.plotly_chart(
            feature_importance_chart(names, importances, top_n=top_n),
            width="stretch",
            config={"displayModeBar": False},
        )
