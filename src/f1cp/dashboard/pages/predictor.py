"""Ranked championship probabilities for one round of one season."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict
from f1cp.dashboard.components.charts import probability_bar_chart
from f1cp.dashboard.components.layout import (
    control_panel,
    model_version_selector,
    season_round_inputs,
    show_api_error,
)
from f1cp.dashboard.components.tables import (
    data_table,
    prediction_columns,
    prediction_rows_frame,
)
from f1cp.dashboard.components.theme import Stat, card, page_header, section, stat_grid

if TYPE_CHECKING:
    import pandas as pd

    from f1cp.dashboard.api_client import DashboardApiClient

_TITLE = "Title odds"
_SUBTITLE = "Every driver's probability of winning the drivers' championship, as of a chosen round."


def render(client: DashboardApiClient) -> None:
    """Render the title-odds page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Prediction")
        season, round_number = season_round_inputs("predictor")
        model_version_id = model_version_selector(client, "predictor")

    with st.spinner("Scoring the field..."):
        try:
            result = cached_predict(
                client, season=season, round_number=round_number, model_version_id=model_version_id
            )
        except ApiError as error:
            page_header(_TITLE, _SUBTITLE)
            show_api_error(error)
            return

    page_header(
        _TITLE,
        _SUBTITLE,
        meta=[
            (f"{result['season']} · Round {result['round']}", "accent"),
            (f"Scored by {result['model_name']}", ""),
        ],
    )

    if not result["is_out_of_sample"]:
        st.warning(
            "This model was fitted on data that includes this season, so these odds are a "
            "replay of a result it already knows -- they look far stronger than the model's "
            "real skill. See the Model leaderboard page for the honest, walk-forward number.",
            icon=":material/history:",
        )

    ranked = sorted(result["drivers"], key=lambda driver: driver["rank"])
    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    top_three = sum(driver["probability"] for driver in ranked[:3])

    stat_grid(
        [
            Stat(
                label="Favourite",
                value=leader["driver_name"],
                detail=f"{leader['probability']:.1%} to take the title",
                tone="accent",
            ),
            Stat(
                label="Lead over second",
                value=(
                    f"{leader['probability'] - runner_up['probability']:.1%}" if runner_up else "-"
                ),
                detail=f"over {runner_up['driver_name']}" if runner_up else None,
            ),
            Stat(
                label="Top three combined",
                value=f"{top_three:.1%}",
                detail="share of the championship between them",
            ),
            Stat(
                label="Drivers scored",
                value=str(len(ranked)),
                detail="contesting this round",
            ),
        ]
    )

    frame: pd.DataFrame = prediction_rows_frame(result["drivers"])

    section(
        "The leading contenders",
        "The strongest twelve. Every driver scored this round is in the table below.",
    )
    with card():
        st.plotly_chart(
            probability_bar_chart(frame, value_col="Title chance", label_col="Driver"),
            width="stretch",
            config={"displayModeBar": False},
        )

    section(
        "The full field",
        "Calibrated is the model's probability before rescaling across the field; raw is "
        "its output before calibration. Sort by any column.",
    )
    data_table(frame, prediction_columns())
