"""Shared controls: the sidebar inputs every page filters with, and status feedback.

Controls live in the sidebar and content lives in the main column, without exception --
a reader should never have to look for where a filter went. Each builder here takes a
``key_prefix`` so the same control can appear on more than one page in a single run
without colliding on widget state.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.components.theme import inject_theme

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

__all__ = [
    "control_panel",
    "model_version_selector",
    "season_round_inputs",
    "season_selector",
    "show_api_error",
    "status_badge",
]

# Not a widget key: see the note in `model_version_selector` about surviving navigation.
_MODEL_CHOICE_KEY = "_f1cp_model_choice"


def control_panel(title: str = "Filters") -> None:
    """Open the sidebar's control section with a consistent heading.

    Args:
        title: What the controls below it change.
    """
    inject_theme()
    st.sidebar.markdown(
        f'<p class="f1-card__title" style="margin:0 0 0.75rem">{title}</p>',
        unsafe_allow_html=True,
    )


def season_selector(key_prefix: str, *, label: str = "Season") -> int:
    """Render a season picker.

    Args:
        key_prefix: Prefix for the widget key.
        label: The control's label.

    Returns:
        The chosen season.
    """
    return int(
        st.number_input(
            label,
            min_value=1950,
            max_value=dt.date.today().year + 1,
            value=dt.date.today().year,
            step=1,
            key=f"{key_prefix}_season_only",
        )
    )


def season_round_inputs(
    key_prefix: str, *, default_season: int | None = None
) -> tuple[int | None, int | None]:
    """Render a "latest round" toggle plus season/round inputs.

    Args:
        key_prefix: Prefix for Streamlit widget keys, so the same widgets can appear on
            more than one page in a single run without key collisions.
        default_season: Season to prefill the number input with when the user switches off
            "latest round". Falls back to the current year.

    Returns:
        ``(season, round)``, both ``None`` when the toggle is on -- the API resolves that
        to the latest race with stored results.
    """
    use_latest = st.toggle("Latest completed round", value=True, key=f"{key_prefix}_use_latest")
    if use_latest:
        return None, None

    season = st.number_input(
        "Season",
        min_value=1950,
        max_value=dt.date.today().year + 1,
        value=default_season or dt.date.today().year,
        step=1,
        key=f"{key_prefix}_season",
    )
    round_number = st.number_input(
        "Round", min_value=1, max_value=40, value=1, step=1, key=f"{key_prefix}_round"
    )
    return int(season), int(round_number)


def model_version_selector(client: DashboardApiClient, key_prefix: str) -> int | None:
    """Render a model-version picker, defaulting to whichever model is live.

    Only offers versions from the *latest* training run: every ``f1cp train run`` adds a
    fresh candidate per zoo algorithm rather than overwriting the last one (see
    ``pages/model_performance.py``), so listing every version ever trained here would make
    this control grow without bound across repeated training runs.

    Args:
        client: The API client.
        key_prefix: Prefix for the widget key.

    Returns:
        The chosen model version id, or ``None`` for "use the live model".
    """
    versions = client.model_versions()
    if not versions:
        return None

    latest_run_id = max(version["training_run_id"] for version in versions)
    latest_versions = [v for v in versions if v["training_run_id"] == latest_run_id]
    live_name = next((v["name"] for v in versions if v["is_active"]), None)

    serving = f"Serving model ({live_name})" if live_name else "Serving model"
    labels: dict[str, int | None] = {serving: None}
    for version in latest_versions:
        labels[version["name"]] = version["id"]

    # Streamlit discards a widget's state as soon as a run does not render it, so simply
    # keying the selectbox is not enough: navigating to another page and back would silently
    # reset the choice to the serving model, which reads as "my selection was ignored".
    # Mirroring it into a key of our own -- one the framework does not manage -- is what
    # makes the choice survive navigation.
    remembered = st.session_state.get(_MODEL_CHOICE_KEY)
    options = list(labels)
    index = options.index(remembered) if remembered in labels else 0

    choice = st.selectbox(
        "Model",
        options,
        index=index,
        key=f"{key_prefix}_model",
        help="Score this page with a different candidate from the latest training run. "
        "This does not change which model the API serves -- activate a version from the "
        "Model leaderboard to do that.",
    )
    st.session_state[_MODEL_CHOICE_KEY] = choice
    return labels[choice]


def status_badge(ok: bool, label: str) -> None:
    """Render a status indicator that never relies on colour alone.

    Args:
        ok: Whether the status is healthy.
        label: Text shown inside the badge.
    """
    inject_theme()
    variant = "ok" if ok else "bad"
    state = "operational" if ok else "unavailable"
    st.markdown(
        f'<span class="f1-chip f1-chip--{variant}"><span class="f1-chip__dot"></span>'
        f"{label}</span><span style='position:absolute;left:-9999px'>{state}</span>",
        unsafe_allow_html=True,
    )


def show_api_error(error: Exception) -> None:
    """Render a consistent error banner for a failed API call.

    Args:
        error: The exception raised by :mod:`f1cp.dashboard.api_client`.
    """
    st.error(str(error), icon=":material/error:")
