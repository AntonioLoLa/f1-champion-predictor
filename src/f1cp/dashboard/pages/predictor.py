"""Ranked championship probabilities for one round of one season."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict
from f1cp.dashboard.components.charts import probability_bar_chart
from f1cp.dashboard.components.labels import model_label, percent
from f1cp.dashboard.components.layout import (
    control_panel,
    in_sample_notice,
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
        season, round_number = season_round_inputs(client, "predictor")
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
            (f"Scored by {model_label(result['model_name'])}", ""),
        ],
    )

    if not result["is_out_of_sample"]:
        in_sample_notice()

    ranked = sorted(result["drivers"], key=lambda driver: driver["rank"])
    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    top_three = sum(driver["probability"] for driver in ranked[:3])

    stat_grid(
        [
            Stat(
                label="Favourite",
                value=leader["driver_name"],
                detail=f"{percent(leader['probability'])} to take the title",
                tone="accent",
            ),
            Stat(
                label="Closest challenger",
                value=runner_up["driver_name"] if runner_up else "-",
                detail=(
                    f"{percent(runner_up['probability'])} · "
                    f"P{runner_up['standing_position']}, {runner_up['points']:.0f} pts"
                    if runner_up and runner_up["standing_position"]
                    else None
                ),
            ),
            Stat(
                label="Top three combined",
                value=percent(top_three),
                detail="share of the title between them",
            ),
            Stat(
                label="Drivers rated",
                value=str(len(ranked)),
                detail="everyone with a place in the standings",
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
        "Every driver's title chance next to where they stand in the championship. Drivers "
        "with equal odds are listed in championship order. Sort by any column.",
    )
    data_table(frame, prediction_columns())

    with st.expander("Model details"):
        st.caption(
            "How the headline odds are made. The model rates each driver on their own; "
            "those ratings are adjusted to match how often similar odds came true in past "
            "seasons, then rescaled so the whole field shares exactly 100%. A driver who "
            "can no longer win on points is set to 0% before that rescale."
        )
        data_table(prediction_rows_frame(result["drivers"], diagnostics=True), prediction_columns())
