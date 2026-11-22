"""Which drivers can still mathematically win the championship."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.components.layout import control_panel, season_selector, show_api_error
from f1cp.dashboard.components.tables import contention_columns, contention_frame, data_table
from f1cp.dashboard.components.theme import Stat, page_header, section, stat_grid

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_SUBTITLE = (
    "Who is still mathematically alive in the title race -- decided by arithmetic over "
    "the remaining calendar, with no model involved."
)


def _rounds_left(count: int) -> str:
    """``"3 rounds to go"``, or that the season is over."""
    if count == 0:
        return "Season complete"
    return f"{count} round{'s' if count != 1 else ''} to go"


def render(client: DashboardApiClient) -> None:
    """Render the elimination page.

    Args:
        client: The API client.
    """
    with st.sidebar:
        control_panel("Season")
        season = season_selector(client, "contention")

    with st.spinner("Working through the remaining calendar..."):
        try:
            result = client.title_contention(season)
        except ApiError as error:
            page_header("Elimination", _SUBTITLE)
            show_api_error(error)
            return

    contenders = result["contenders"]
    alive = [row for row in contenders if row["still_alive"]]
    self_sufficient = [row for row in contenders if row["controls_own_destiny"]]
    leader = max(contenders, key=lambda row: row["current_points"]) if contenders else None

    page_header(
        "Elimination",
        _SUBTITLE,
        meta=[
            (f"{result['season']} · after round {result['round']}", "accent"),
            (_rounds_left(int(result.get("rounds_remaining", 0))), ""),
        ],
    )

    stat_grid(
        [
            Stat(
                label="Can still win",
                value=f"{len(alive)} of {len(contenders)}",
                detail="with everything still to race",
                tone="accent",
            ),
            Stat(
                label="Title in own hands",
                value=str(len(self_sufficient)),
                detail="winning every race left would be enough",
            ),
            Stat(
                label="Eliminated",
                value=str(len(contenders) - len(alive)),
                detail="can no longer catch the leader's points",
                tone="negative" if len(contenders) > len(alive) else "neutral",
            ),
            Stat(
                label="Championship leader",
                value=leader["driver_name"] if leader else "-",
                detail=f"{leader['current_points']:.0f} points" if leader else None,
            ),
        ]
    )

    section(
        "Every driver's ceiling",
        "Max possible is the points total a driver would hold after winning every race and "
        "sprint still to come. The two tick columns answer different questions -- hover "
        "either header for which.",
    )
    data_table(contention_frame(contenders), contention_columns())

    with st.expander("How this is decided"):
        st.markdown(
            "**Can still win** is elimination in the strict sense: a driver is out "
            "only once some rival *already* holds more points than this driver could "
            "reach by winning every race, sprint and fastest lap left. A rival's points "
            "can only go up, and nothing prevents a rival from scoring nothing for the "
            "rest of the season, so their current total is the only floor that exists.\n\n"
            "**In own hands** is the stronger question: would winning out be enough "
            "*even if* the nearest rival took the runner-up spot in every remaining round? "
            "Failing this is not elimination -- it means the title is no longer in the "
            "driver's own hands, and they need a rival to drop points somewhere.\n\n"
            "Both are closed-form: remaining rounds are independent of one another, so "
            "each question has a single optimal scenario and no search over race-by-race "
            "outcomes is needed. Ties go to the contender, since the championship would "
            "then be decided on countback, which this does not model."
        )
