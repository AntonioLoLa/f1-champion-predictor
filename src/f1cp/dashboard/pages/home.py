"""The landing page: where the title race stands, as of the last completed round."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.cache import cached_predict
from f1cp.dashboard.components.labels import model_label, percent, team_name
from f1cp.dashboard.components.layout import in_sample_notice, show_api_error
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


def _standing(driver: dict[str, Any]) -> str:
    """``"P2 · 279 pts"``: where a driver stands in the real championship."""
    position = driver.get("standing_position")
    points = f"{driver['points']:.0f} pts"
    return f"P{position} · {points}" if position else points


def _decided_stats(
    champion: dict[str, Any], runner_up: dict[str, Any] | None, rounds_remaining: int
) -> list[Stat]:
    """The headline once the title is mathematically settled.

    A "100%" card says the same thing less clearly, and invites the question of how sure
    the model is -- when the answer is that no model is involved any more.
    """
    stats = [
        Stat(
            label="Champion",
            value=champion["driver_name"],
            detail=f"Title secured · {team_name(champion) or 'Team unknown'}",
            tone="accent",
        )
    ]
    if runner_up is not None:
        stats.append(
            Stat(
                label="Winning margin",
                value=f"{champion['points'] - runner_up['points']:.0f} pts",
                detail=f"over {runner_up['driver_name']}, {_standing(runner_up)}",
            )
        )
    stats.append(
        Stat(
            label="Rounds to spare",
            value=str(rounds_remaining),
            detail=(
                "still to run, with the title already decided"
                if rounds_remaining
                else "settled at the final round"
            ),
        )
    )
    return stats


def _open_stats(
    leader: dict[str, Any], runner_up: dict[str, Any] | None, contenders: int | None, field: int
) -> list[Stat]:
    """The headline while the title can still go more than one way."""
    stats = [
        Stat(
            label="Favourite",
            value=leader["driver_name"],
            detail=f"{team_name(leader) or 'Team unknown'} · {_standing(leader)}",
            tone="accent",
        ),
        Stat(
            label="Title chance",
            value=percent(leader["probability"]),
            # Both numbers side by side rather than their difference: a gap between two
            # probabilities is in percentage points, and "+40%" reads as a relative change.
            detail=(
                f"vs {percent(runner_up['probability'])} for {runner_up['driver_name']}"
                if runner_up
                else None
            ),
        ),
    ]
    if runner_up:
        stats.append(
            Stat(
                label="Closest challenger",
                value=runner_up["driver_name"],
                detail=f"{percent(runner_up['probability'])} · {_standing(runner_up)}",
            )
        )
    if contenders is not None:
        stats.append(
            Stat(
                label="Can still win",
                value=str(contenders),
                detail=f"of {field} drivers, on points alone",
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
    rounds_remaining = 0
    try:
        contention = client.title_contention(latest["season"], round_number=latest["round"])
    except ApiError:
        pass
    else:
        contenders = len([row for row in contention["contenders"] if row["still_alive"]])
        rounds_remaining = int(contention.get("rounds_remaining", 0))

    page_header(
        _TITLE,
        _SUBTITLE,
        meta=[
            (f"{latest['season']} · Round {latest['round']}", "accent"),
            (model_label(latest["model_name"]), "ok" if health["database"] else "bad"),
        ],
    )

    if not latest["is_out_of_sample"]:
        in_sample_notice()

    ranked = sorted(latest["drivers"], key=lambda driver: driver["rank"])
    leader = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    # Decided means arithmetic, not the model: one driver left who can still win. The
    # served probabilities are then 100% and 0% by construction (eliminated drivers are
    # zeroed before the rescale), so they are the least informative thing to lead with.
    decided = contenders == 1
    stat_grid(
        _decided_stats(leader, runner_up, rounds_remaining)
        if decided
        else _open_stats(leader, runner_up, contenders, len(ranked))
    )

    section(
        "Who the model favours",
        (
            "The title is decided, so every other driver is at 0% by arithmetic. Drivers "
            "with equal odds are listed in championship order."
            if decided
            else "Each driver's share of one championship: the odds add up to 100% across "
            "the field. Drivers with equal odds are listed in championship order."
        ),
    )

    left, right = st.columns([1, 1], gap="large")

    with left, card("Title odds"):
        leaderboard(
            [
                BoardRow(
                    position=str(driver["rank"]),
                    name=driver["driver_name"],
                    secondary=team_name(driver),
                    value=percent(driver["probability"]),
                    share=driver["probability"],
                )
                for driver in ranked[:_BOARD_SIZE]
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
