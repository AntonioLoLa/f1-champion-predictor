"""How one driver's championship probability moved across a season."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict_season
from f1cp.dashboard.components.charts import driver_highlight_trajectory_chart
from f1cp.dashboard.components.labels import percent
from f1cp.dashboard.components.layout import (
    control_panel,
    in_sample_notice,
    season_selector,
    show_api_error,
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

_SUBTITLE = "One driver's title probability round by round, against the rest of the field."


def _long_frame(rounds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "round": round_result["round"],
            "driver_id": driver["driver_id"],
            "driver_name": driver["driver_name"],
            "probability": driver["probability"],
            "rank": driver["rank"],
            "standing_position": driver.get("standing_position"),
        }
        for round_result in rounds
        for driver in round_result["drivers"]
    ]
    return pd.DataFrame(rows)


def render(client: DashboardApiClient) -> None:
    """Render the driver trajectory page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Season")
        season = season_selector(client, "driver_analysis")

    with st.spinner("Loading the season..."):
        try:
            rounds = cached_predict_season(client, season)
        except ApiError as error:
            page_header("Driver trajectory", _SUBTITLE)
            show_api_error(error)
            return

    if not rounds:
        page_header("Driver trajectory", _SUBTITLE)
        empty_state(
            f"Nothing stored for {season}",
            "Pick another season, or ingest this one from the Pipeline page.",
        )
        return

    frame = _long_frame(rounds)
    # Ordered by the championship table at the last scored round rather than alphabetically
    # or by the model's odds, so the page opens on the championship leader and the list
    # reads like the standings a fan already knows -- odds would tie a decided season's
    # whole field at 0% and fall back to an arbitrary order.
    final_round = int(frame["round"].max())
    ranked = frame[frame["round"] == final_round].sort_values(
        ["standing_position", "rank"], na_position="last"
    )
    remainder = frame[~frame["driver_id"].isin(ranked["driver_id"])]
    ordered = pd.concat([ranked, remainder.sort_values("driver_name")])
    drivers = ordered[["driver_id", "driver_name"]].drop_duplicates()
    labels = dict(zip(drivers["driver_name"], drivers["driver_id"], strict=True))

    with st.sidebar:
        chosen_name = st.selectbox("Driver", list(labels), key="driver_analysis_driver")
    chosen_id = labels[chosen_name]

    history = frame[frame["driver_id"] == chosen_id].sort_values("round")
    latest_round = int(history["round"].max())
    latest = float(history[history["round"] == latest_round]["probability"].iloc[0])
    opening = float(history["probability"].iloc[0])
    peak = float(history["probability"].max())

    page_header(
        "Driver trajectory",
        _SUBTITLE,
        meta=[(chosen_name, "accent"), (f"{season} · {len(rounds)} rounds", "")],
    )

    if not rounds[0]["is_out_of_sample"]:
        in_sample_notice()

    with st.spinner("Loading standings..."):
        try:
            standings = client.standings(season, round_number=latest_round)
        except ApiError:
            standing = None
        else:
            standing = next(
                (row for row in standings["standings"] if row["driver_id"] == chosen_id), None
            )

    change = latest - opening
    stats = [
        Stat(
            label=f"After round {latest_round}",
            value=percent(latest),
            # The starting value, not the difference: a gap between two probabilities is
            # in percentage points, and "+12%" reads as a relative change.
            detail=f"from {percent(opening)} after round {int(history['round'].iloc[0])}",
            tone="positive" if change > 0 else "negative" if change < 0 else "neutral",
        ),
        Stat(label="Season peak", value=percent(peak), detail="highest of the year"),
    ]
    if standing is not None:
        stats.append(
            Stat(
                label="Championship position",
                value=f"P{standing['position']}",
                detail=f"{standing['points']:.0f} points, {standing['wins']:.0f} wins",
            )
        )
    stat_grid(stats)

    section(
        f"{chosen_name} against the field",
        "Every other driver is drawn behind in gray, so the highlighted line can be read "
        "against the whole championship rather than in isolation.",
    )
    with card():
        st.plotly_chart(
            driver_highlight_trajectory_chart(
                frame,
                x_col="round",
                y_col="probability",
                group_col="driver_id",
                label_col="driver_name",
                highlight_id=chosen_id,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
