"""A season replayed round by round: how the title race actually unfolded."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict_season
from f1cp.dashboard.components.charts import multi_driver_trajectory_chart
from f1cp.dashboard.components.layout import control_panel, season_selector, show_api_error
from f1cp.dashboard.components.tables import (
    data_table,
    prediction_columns,
    prediction_rows_frame,
    standings_columns,
    standings_frame,
)
from f1cp.dashboard.components.theme import (
    Stat,
    card,
    empty_state,
    page_header,
    section,
    stat_grid,
)

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_MAX_SERIES = 8
_SUBTITLE = "Replay a completed season and watch the model's view of the title change race by race."


def _long_frame(rounds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "round": round_result["round"],
            "driver_id": driver["driver_id"],
            "driver_name": driver["driver_name"],
            "probability": driver["probability"],
        }
        for round_result in rounds
        for driver in round_result["drivers"]
    ]
    return pd.DataFrame(rows)


def render(client: DashboardApiClient) -> None:
    """Render the season replay page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Season")
        season = season_selector("historical")

    with st.spinner("Loading the season..."):
        try:
            rounds = cached_predict_season(client, season)
        except ApiError as error:
            page_header("Season replay", _SUBTITLE)
            show_api_error(error)
            return

    if not rounds:
        page_header("Season replay", _SUBTITLE)
        empty_state(
            f"Nothing stored for {season}",
            "Pick another season, or ingest this one from the Pipeline page.",
        )
        return

    frame = _long_frame(rounds)
    final_round = int(frame["round"].max())
    final_ranking = frame[frame["round"] == final_round].sort_values("probability", ascending=False)
    winner = final_ranking.iloc[0]

    page_header(
        "Season replay",
        _SUBTITLE,
        meta=[(f"{season} · {len(rounds)} rounds scored", "accent")],
    )

    if not rounds[0]["is_out_of_sample"]:
        st.warning(
            "The active model was fitted on data that includes this season, so this replay "
            "shows a result it already knows -- it looks far stronger than the model's real "
            "skill. See the Model leaderboard page for the honest, walk-forward number.",
            icon=":material/history:",
        )

    # How early the model settled on its eventual pick: the first round after which it
    # never changed its mind again.
    leaders = frame.loc[frame.groupby("round")["probability"].idxmax()].sort_values("round")
    settled_from = final_round
    for _, row in leaders[::-1].iterrows():
        if row["driver_id"] != winner["driver_id"]:
            break
        settled_from = int(row["round"])

    stat_grid(
        [
            Stat(
                label=f"Model's pick after round {final_round}",
                value=str(winner["driver_name"]),
                detail=f"{winner['probability']:.1%} at the final scored round",
                tone="accent",
            ),
            Stat(
                label="Settled from round",
                value=str(settled_from),
                detail="never changed its favourite after this",
            ),
            Stat(
                label="Lead changes",
                value=str(int((leaders["driver_id"] != leaders["driver_id"].shift()).sum() - 1)),
                detail="times the favourite changed hands",
            ),
        ]
    )

    section(
        "How the title race moved",
        f"Up to {_MAX_SERIES} drivers at a time -- past that, colours stop being reliably "
        "distinguishable. Choose who to plot in the sidebar.",
    )

    default_ids = final_ranking["driver_id"].head(6).tolist()
    labels = dict(zip(final_ranking["driver_name"], final_ranking["driver_id"], strict=True))
    default_names = [name for name, driver_id in labels.items() if driver_id in default_ids]

    with st.sidebar:
        chosen_names = st.multiselect(
            "Drivers to plot",
            list(labels),
            default=default_names,
            max_selections=_MAX_SERIES,
            key="historical_drivers",
        )

    chosen_ids = [labels[name] for name in chosen_names] or default_ids
    with card():
        st.plotly_chart(
            multi_driver_trajectory_chart(
                frame[frame["driver_id"].isin(chosen_ids)],
                x_col="round",
                y_col="probability",
                group_col="driver_id",
                label_col="driver_name",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    section(
        "Standings at any point in the season",
        "Drag to a round to see the real championship table as it stood after it.",
    )
    selected_round = st.slider(
        "Round",
        min_value=1,
        max_value=final_round,
        value=final_round,
        key="standings_round",
        label_visibility="collapsed",
    )
    with st.spinner("Loading standings..."):
        try:
            standings = client.standings(season, round_number=selected_round)
        except ApiError as error:
            show_api_error(error)
        else:
            data_table(standings_frame(standings["standings"]), standings_columns(), height=420)

    section(
        f"What the model predicted after round {final_round}",
        "Its final ranking of the season, for comparison against the standings above.",
    )
    last_round_response = next(r for r in rounds if r["round"] == final_round)
    data_table(prediction_rows_frame(last_round_response["drivers"]), prediction_columns())
