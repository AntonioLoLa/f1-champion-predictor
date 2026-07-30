"""The results archive: what actually happened, season by season and race by race.

The only page in the app with no model on it. Everything here is the ingested record, and
that is the point: it is the reference the prediction pages are read against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.components.layout import control_panel, show_api_error
from f1cp.dashboard.components.tables import (
    constructor_standings_columns,
    constructor_standings_frame,
    data_table,
    race_calendar_columns,
    race_calendar_frame,
    race_results_columns,
    race_results_frame,
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

_TITLE = "Results archive"
_SUBTITLE = (
    "The championship as it actually finished -- every season, every round, every "
    "classification. No model involved."
)
_TABLE_HEIGHT = 460


def _season_stats(summary: dict[str, Any], races: list[dict[str, Any]]) -> list[Stat]:
    """The four numbers that describe a season at a glance."""
    run = [race for race in races if race["winner_id"]]
    distinct_winners = len({race["winner_id"] for race in run})
    sprints = len([race for race in races if race["has_sprint"]])

    stats = [
        Stat(
            label="Champion",
            value=summary["champion_name"] or "Season in progress",
            detail=f"{summary['rounds_completed']} of {summary['rounds']} rounds run",
            tone="accent",
        ),
        Stat(label="Rounds on the calendar", value=str(summary["rounds"])),
        Stat(
            label="Different winners",
            value=str(distinct_winners),
            detail=f"across {len(run)} races run",
        ),
    ]
    if sprints:
        stats.append(Stat(label="Sprint weekends", value=str(sprints)))
    return stats


def render(client: DashboardApiClient) -> None:
    """Render the results archive.

    Args:
        client: The API client.
    """
    with st.spinner("Loading seasons..."):
        try:
            catalogue = client.seasons()["seasons"]
        except ApiError as error:
            page_header(_TITLE, _SUBTITLE)
            show_api_error(error)
            return

    if not catalogue:
        page_header(_TITLE, _SUBTITLE)
        empty_state(
            "Nothing ingested yet",
            "Run a data update from the Pipeline page and the archive fills in from the "
            "sources.",
        )
        return

    summaries = {int(row["season"]): row for row in catalogue}
    with st.sidebar:
        control_panel("Archive")
        season = st.selectbox(
            "Season",
            list(summaries),
            key="archive_season",
            help="Every season with ingested data, newest first.",
        )

    with st.spinner("Loading the calendar..."):
        try:
            calendar = client.season_races(season)["races"]
        except ApiError as error:
            page_header(_TITLE, _SUBTITLE)
            show_api_error(error)
            return

    summary = summaries[season]
    page_header(
        _TITLE,
        _SUBTITLE,
        meta=[
            (f"{season} season", "accent"),
            (summary["champion_name"] or "In progress", "ok" if summary["champion_name"] else ""),
        ],
    )

    stat_grid(_season_stats(summary, calendar))

    section(
        "The season's calendar",
        "Every round with its circuit and, where the race has been run, its winner.",
    )
    data_table(race_calendar_frame(calendar), race_calendar_columns(), height=_TABLE_HEIGHT)

    run = [race for race in calendar if race["winner_id"]]
    if not run:
        empty_state(
            f"{season} has not been run yet",
            "The calendar is published, but no results have been ingested for it.",
        )
        return

    section(
        "A single Grand Prix",
        "The full classification, winner first and retirements last, exactly as the race "
        "was classified.",
    )
    labels = {f"Round {race['round']} · {race['name']}": race["round"] for race in run}
    chosen = st.selectbox("Grand Prix", list(labels), index=len(labels) - 1)
    with st.spinner("Loading the classification..."):
        try:
            results = client.race_results(season, labels[chosen])
        except ApiError as error:
            show_api_error(error)
        else:
            data_table(race_results_frame(results), race_results_columns())

    section(
        "Championship standings",
        f"Both championships as they stood after round {labels[chosen]}.",
    )
    drivers_column, constructors_column = st.columns(2, gap="large")

    with drivers_column, card("Drivers"), st.spinner("Loading driver standings..."):
        try:
            standings = client.standings(season, round_number=labels[chosen])
        except ApiError as error:
            show_api_error(error)
        else:
            data_table(
                standings_frame(standings["standings"]), standings_columns(), height=_TABLE_HEIGHT
            )

    with constructors_column, card("Constructors"), st.spinner("Loading constructor standings..."):
        try:
            constructors = client.constructor_standings(season, round_number=labels[chosen])
        except ApiError as error:
            show_api_error(error)
        else:
            if constructors:
                data_table(
                    constructor_standings_frame(constructors),
                    constructor_standings_columns(),
                    height=_TABLE_HEIGHT,
                )
            else:
                # The constructors' championship only began in 1958, and older seasons in
                # the archive predate it.
                st.caption("No constructors' championship recorded for this round.")
