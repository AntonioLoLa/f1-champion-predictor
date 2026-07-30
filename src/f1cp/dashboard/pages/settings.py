"""Pipeline controls: ingest data, start a training run, check on what is running."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from f1cp.config import load_settings
from f1cp.dashboard.api_client import ApiError
from f1cp.dashboard.components.layout import show_api_error
from f1cp.dashboard.components.theme import Stat, card, page_header, section, stat_grid

if TYPE_CHECKING:
    from f1cp.dashboard.api_client import DashboardApiClient

_SUBTITLE = "Run the data and training pipeline, and check on what the service is doing."


def render(client: DashboardApiClient) -> None:
    """Render the pipeline page.

    Args:
        client: The API client.
    """
    dashboard_settings = load_settings().dashboard

    with st.spinner("Checking service status..."):
        try:
            health = client.health()
        except ApiError as error:
            page_header("Pipeline", _SUBTITLE)
            show_api_error(error)
            return

    page_header(
        "Pipeline",
        _SUBTITLE,
        meta=[
            ("API " + health["status"], "ok" if health["status"] == "ok" else "bad"),
            ("Database", "ok" if health["database"] else "bad"),
        ],
    )

    stat_grid(
        [
            Stat(
                label="Serving model",
                value=health.get("active_model") or "None",
                detail="what predictions are scored with",
                tone="accent",
            ),
            Stat(
                label="API endpoint",
                value=dashboard_settings.api_base_url.replace("http://", ""),
                detail=f"{dashboard_settings.request_timeout_seconds:.0f}s request timeout",
            ),
        ]
    )

    section(
        "Ingest race data",
        "Mirrors `f1cp data update`. A full bootstrap walks decades of results and takes a "
        "long time -- leave it running in the background.",
    )
    with card(), st.form("update_data_form"):
        left, right = st.columns(2)
        with left:
            bootstrap = st.checkbox("Bootstrap full history")
            refresh = st.checkbox("Re-fetch instead of updating incrementally")
            background = st.checkbox("Run in the background", value=True)
        with right:
            seasons_raw = st.text_input(
                "Seasons",
                placeholder="2024, 2025 -- blank for the current season",
            )
            min_season = st.number_input(
                "Earliest season to bootstrap", min_value=1950, value=1981, step=1
            )
        submitted = st.form_submit_button("Run data update", type="primary")

    if submitted:
        seasons = [int(value) for value in seasons_raw.split(",") if value.strip()] or None
        with st.spinner("Requesting data update..."):
            try:
                response = client.update_data(
                    bootstrap=bootstrap,
                    seasons=seasons,
                    refresh=refresh,
                    min_season=int(min_season) if bootstrap else None,
                    background=background,
                )
            except ApiError as error:
                show_api_error(error)
            else:
                st.success(f"Update {response['status']}.", icon=":material/check_circle:")
                if response["reports"]:
                    with st.expander("Ingestion report"):
                        st.json(response["reports"])

    section(
        "Train the model zoo",
        "Walks forward over completed seasons, fits every candidate, calibrates, scores "
        "and registers a winner.",
    )
    with card():
        if st.button("Start training run", type="primary"):
            with st.spinner("Scheduling training run..."):
                try:
                    scheduled = client.train()
                except ApiError as error:
                    show_api_error(error)
                else:
                    st.session_state["last_run_uuid"] = scheduled["run_uuid"]
                    st.success(
                        f"Run {scheduled['run_uuid']} scheduled.", icon=":material/check_circle:"
                    )

        check, button = st.columns([3, 1], vertical_alignment="bottom")
        with check:
            run_uuid = st.text_input(
                "Training run id",
                value=st.session_state.get("last_run_uuid", ""),
                placeholder="Paste a run id to check on it",
            )
        with button:
            checking = st.button("Check status", width="stretch")

        if checking and run_uuid:
            with st.spinner("Checking run status..."):
                try:
                    status = client.train_status(run_uuid)
                except ApiError as error:
                    show_api_error(error)
                else:
                    st.json(status)
