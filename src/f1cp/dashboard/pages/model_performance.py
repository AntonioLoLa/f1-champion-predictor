"""The model zoo's scoreboard: what was trained, how it scored, and what is serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.components.charts import metric_comparison_chart
from f1cp.dashboard.components.layout import show_api_error
from f1cp.dashboard.components.tables import (
    data_table,
    model_versions_columns,
    model_versions_frame,
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

_METADATA_COLUMNS = {
    "id",
    "training_run_id",
    "name",
    "algorithm",
    "is_active",
    "is_best",
    "created_at",
}

# Metrics where a smaller number is a better model. Anything else (a hit rate, an
# accuracy) reads the other way, and the winning bar has to follow.
_LOWER_IS_BETTER_HINTS = ("loss", "error", "brier", "mae", "rmse")

_SUBTITLE = (
    "Every candidate from the latest training run, how it scored under walk-forward "
    "validation, and which one the API is serving."
)


def _lower_is_better(metric: str) -> bool:
    return any(hint in metric.lower() for hint in _LOWER_IS_BETTER_HINTS)


def render(client: DashboardApiClient) -> None:
    """Render the model leaderboard page.

    Args:
        client: The API client.
    """
    with st.spinner("Loading trained models..."):
        try:
            versions = client.model_versions()
        except ApiError as error:
            page_header("Model leaderboard", _SUBTITLE)
            show_api_error(error)
            return

    if not versions:
        page_header("Model leaderboard", _SUBTITLE)
        empty_state(
            "No models trained yet",
            "Start a training run from the Pipeline page; every candidate will appear "
            "here with its cross-validated scores.",
        )
        return

    full_frame = model_versions_frame(versions)

    # Every `f1cp train run` adds a fresh set of candidates (one per zoo algorithm) rather
    # than overwriting the last one, so the full table grows without bound across repeated
    # training runs. The main view always shows only the latest run -- no toggle, no
    # state that can be left on by accident -- full history lives in its own collapsed
    # expander below instead.
    latest_run_id = int(full_frame["training_run_id"].max())
    frame = full_frame[full_frame["training_run_id"] == latest_run_id]

    live = next((version for version in versions if version["is_active"]), None)
    best = frame[frame["is_best"]]

    page_header(
        "Model leaderboard",
        _SUBTITLE,
        meta=[
            (f"Run #{latest_run_id}", "accent"),
            (f"{len(frame)} candidates", ""),
        ],
    )

    stat_grid(
        [
            Stat(
                label="Serving now",
                value=live["name"] if live else "None",
                detail=live["algorithm"] if live else "No model activated",
                tone="accent",
            ),
            Stat(
                label="Best of this run",
                value=str(best.iloc[0]["name"]) if len(best) else "-",
                detail="selected by the training pipeline",
            ),
            Stat(
                label="Candidates trained",
                value=str(len(frame)),
                detail=f"{len(full_frame)} across every run",
            ),
        ]
    )

    section(
        "How the candidates compared",
        "Scores come from walk-forward validation over completed seasons -- each model is "
        "judged only on seasons it never saw during fitting.",
    )

    metric_columns = sorted(column for column in frame.columns if column not in _METADATA_COLUMNS)
    if metric_columns:
        metric = st.selectbox(
            "Metric",
            metric_columns,
            index=0,
            help="Lower is better for losses and errors; higher is better for hit rates.",
        )
        comparable = frame.dropna(subset=[metric])
        if len(comparable):
            with card():
                st.plotly_chart(
                    metric_comparison_chart(
                        comparable,
                        metric_col=metric,
                        category_col="name",
                        lower_is_better=_lower_is_better(metric),
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                )

    section("Every candidate in this run", "Sort by any metric to rank them yourself.")
    data_table(frame, model_versions_columns(frame))

    with st.expander(f"Full training history · {len(full_frame)} versions across every run"):
        data_table(full_frame, model_versions_columns(full_frame), height=360)

    section(
        "Change the serving model",
        "Activating a version switches what every prediction in this app -- and every API "
        "caller -- is scored with.",
    )
    visible_ids = set(frame["id"])
    labels = {
        f"{version['name']}{' · live' if version['is_active'] else ''}": version["id"]
        for version in versions
        if version["id"] in visible_ids
    }
    choose, act = st.columns([3, 1], vertical_alignment="bottom")
    with choose:
        choice = st.selectbox("Model version", list(labels), label_visibility="collapsed")
    with act:
        activate = st.button("Activate", type="primary", width="stretch")

    if activate:
        with st.spinner("Activating..."):
            try:
                client.activate_model(labels[choice])
            except ApiError as error:
                show_api_error(error)
            else:
                st.success(f"{choice} is now serving predictions.", icon=":material/check_circle:")
                st.rerun()
