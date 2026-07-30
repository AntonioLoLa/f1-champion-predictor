"""The landing page: where the title race stands, as of the last completed round."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict
from f1cp.dashboard.components.layout import show_api_error
from f1cp.dashboard.components.tables import (
    data_table,
    standings_columns,
    standings_frame,
)
from f1cp.dashboard.components.theme import (
    BoardRow,
    Stat,
    card,
    empty_state,
    leaderboard,
    page_header,
    section,
    stat_grid,
)

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_BOARD_SIZE = 8
# Matches the height of the eight-row odds panel beside it, so the two cards align.
_STANDINGS_HEIGHT = 460
_TITLE = "Championship overview"
_SUBTITLE = "Where the Formula 1 drivers' title race stands after the last completed Grand Prix."


def _headline_stats(drivers: list[dict[str, Any]], contenders: int | None) -> list[Stat]:
    """The four numbers that answer "what is the state of the title race"."""
    ranked = sorted(drivers, key=lambda driver: driver["rank"])
    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = (
        leader["probability"] - runner_up["probability"] if runner_up else leader["probability"]
    )

    stats = [
        Stat(
            label="Favourite",
            value=leader["driver_name"],
            detail=leader["constructor_id"] or "Team unknown",
            tone="accent",
        ),
        Stat(
            label="Title chance",
            value=f"{leader['probability']:.1%}",
            detail=f"{margin:+.1%} vs {runner_up['driver_name']}" if runner_up else None,
            tone="positive" if margin > 0 else "neutral",
        ),
    ]
    if runner_up:
        stats.append(
            Stat(
                label="Closest challenger",
                value=runner_up["driver_name"],
                detail=f"{runner_up['probability']:.1%} to win the title",
            )
        )
    if contenders is not None:
        stats.append(
            Stat(
                label="Still in contention",
                value=str(contenders),
                detail=f"of {len(drivers)} drivers, mathematically",
            )
        )
    return stats


def render(client: DashboardApiClient) -> None:
    """Render the overview page.

    Args:
        client: The API client.
    """
    with st.spinner("Contacting the prediction service..."):
        try:
            health = client.health()
        except ApiError as error:
            page_header(_TITLE, _SUBTITLE)
            show_api_error(error)
            return

    with st.spinner("Loading the latest prediction..."):
        try:
            latest = cached_predict(client, season=None, round_number=None, model_version_id=None)
        except ApiError:
            latest = None

    if latest is None:
        page_header(
            _TITLE,
            _SUBTITLE,
            meta=[("Service online", "ok" if health["status"] == "ok" else "bad")],
        )
        empty_state(
            "No predictions yet",
            "Ingest a season and train a model from the Pipeline page, then this view "
            "will fill in on its own.",
        )
        return

    # Contention is a nice-to-have on this page, not the point of it: if that endpoint is
    # unavailable the overview still stands on the prediction it already has.
    contenders: int | None = None
    try:
        contention = client.title_contention(latest["season"])
    except ApiError:
        pass
    else:
        contenders = len([row for row in contention["contenders"] if row["still_alive"]])

    page_header(
        _TITLE,
        _SUBTITLE,
        meta=[
            (f"{latest['season']} · Round {latest['round']}", "accent"),
            (latest["model_name"], "ok" if health["database"] else "bad"),
        ],
    )

    stat_grid(_headline_stats(latest["drivers"], contenders))

    section(
        "Who the model favours",
        "Probabilities are rescaled to sum to 100% across the field, so these are shares "
        "of one championship rather than independent guesses.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left, card("Title odds"):
        ranked = sorted(latest["drivers"], key=lambda driver: driver["rank"])[:_BOARD_SIZE]
        leaderboard(
            [
                BoardRow(
                    position=str(driver["rank"]),
                    name=driver["driver_name"],
                    secondary=driver["constructor_id"] or None,
                    value=f"{driver['probability']:.1%}",
                    share=driver["probability"],
                )
                for driver in ranked
            ]
        )

    with right, card("Championship standings"), st.spinner("Loading standings..."):
        try:
            standings = client.standings(latest["season"], round_number=latest["round"])
        except ApiError as error:
            show_api_error(error)
        else:
            # The whole field, not a top-N: the odds panel beside it is already the
            # shortlist, so truncating here would just hide the real table twice. A fixed
            # height keeps it scrolling inside its card instead of stretching the page to
            # match a twenty-plus-driver grid.
            data_table(
                standings_frame(standings["standings"]),
                standings_columns(),
                height=_STANDINGS_HEIGHT,
            )
