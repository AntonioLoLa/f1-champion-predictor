"""Tabular views of API responses, and the column formatting that makes them readable.

Each ``*_frame`` builder pairs with a ``*_columns`` spec, kept side by side in this module
so a renamed column and its formatting can never drift apart. The frames hold real values
-- a probability stays a 0-1 float, points stay numeric -- and formatting happens at the
column level, so Streamlit can still sort numerically on a column the reader sees as
"38.4%".

Every table is rendered through :func:`data_table`, which is the only place that decides
what a table looks like.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from streamlit.elements.lib.column_types import ColumnConfig

__all__ = [
    "constructor_standings_columns",
    "constructor_standings_frame",
    "contention_columns",
    "contention_frame",
    "data_table",
    "model_versions_columns",
    "model_versions_frame",
    "prediction_columns",
    "prediction_rows_frame",
    "race_calendar_columns",
    "race_calendar_frame",
    "race_results_columns",
    "race_results_frame",
    "standings_columns",
    "standings_frame",
]

_METRIC_FORMAT = "%.4f"


def data_table(
    frame: pd.DataFrame,
    columns: dict[str, ColumnConfig] | None = None,
    *,
    height: int | None = None,
) -> None:
    """Render a frame as the app's one table style.

    Args:
        frame: The data to show.
        columns: Per-column formatting, from the matching ``*_columns`` builder.
        height: Fixed pixel height. Set it where a table is long enough to scroll inside
            its own panel rather than pushing the rest of the page down.
    """
    st.dataframe(
        frame,
        column_config=columns,
        hide_index=True,
        width="stretch",
        # "auto" sizes to the content; a pixel height is only passed when the caller
        # explicitly wants the table to scroll inside its own panel.
        height=height if height is not None else "auto",
    )


def prediction_rows_frame(drivers: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per driver's ranked prediction, ready for display.

    Args:
        drivers: The ``drivers`` list from a ``RoundPredictionsResponse``.

    Returns:
        A frame ordered by rank with human-readable column names.
    """
    frame = pd.DataFrame(drivers).sort_values("rank")
    return frame.rename(
        columns={
            "rank": "Rank",
            "driver_name": "Driver",
            "constructor_id": "Team",
            "probability": "Title chance",
            "probability_calibrated": "Calibrated",
            "probability_raw": "Raw",
        }
    )[["Rank", "Driver", "Team", "Title chance", "Calibrated", "Raw"]]


def prediction_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`prediction_rows_frame`.

    The headline probability renders as a bar rather than a number: it is the column the
    reader scans, and a bar is read without arithmetic. The two diagnostic probabilities
    behind it stay as plain percentages -- they are for checking, not scanning.
    """
    return {
        "Rank": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Title chance": st.column_config.ProgressColumn(
            "Title chance",
            help="Probability of winning the championship, rescaled to sum to 100% "
            "across the field.",
            format="percent",
            min_value=0.0,
            max_value=1.0,
        ),
        "Calibrated": st.column_config.NumberColumn(
            "Calibrated",
            help="Model probability after calibration, before rescaling across the field.",
            format="percent",
        ),
        "Raw": st.column_config.NumberColumn(
            "Raw",
            help="Model output before calibration.",
            format="percent",
        ),
    }


def standings_frame(standings: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per driver's championship standing, ready for display.

    Args:
        standings: The ``standings`` list from a ``SeasonStandingsResponse``.

    Returns:
        A frame ordered by position with human-readable column names.
    """
    frame = pd.DataFrame(standings).sort_values("position")
    return frame.rename(
        columns={
            "position": "Pos",
            "driver_name": "Driver",
            "constructor_id": "Team",
            "points": "Points",
            "wins": "Wins",
        }
    )[["Pos", "Driver", "Team", "Points", "Wins"]]


def standings_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`standings_frame`."""
    return {
        "Pos": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Points": st.column_config.NumberColumn("Points", format="%.0f"),
        "Wins": st.column_config.NumberColumn("Wins", format="%d"),
    }


def contention_frame(contenders: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per driver's title-contention status, ready for display.

    Args:
        contenders: The ``contenders`` list from a ``TitleContentionResponse``.

    Returns:
        A frame ordered by current points with human-readable column names. Both status
        columns stay boolean (not text) so they render as native checkbox columns.
    """
    frame = pd.DataFrame(contenders).sort_values("current_points", ascending=False)
    frame = frame.assign(
        eliminated_by=frame.apply(
            lambda row: "-" if row["still_alive"] else (row["eliminated_by_name"] or "-"),
            axis=1,
        ),
    )
    return frame.rename(
        columns={
            "driver_name": "Driver",
            "current_points": "Points",
            "best_case_points": "Maximum possible",
            "still_alive": "In contention",
            "controls_own_destiny": "Needs no help",
            "eliminated_by": "Eliminated by",
        }
    )[
        [
            "Driver",
            "Points",
            "Maximum possible",
            "In contention",
            "Needs no help",
            "Eliminated by",
        ]
    ]


def contention_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`contention_frame`.

    The two status columns answer genuinely different questions and are easy to conflate,
    so each carries its own help text rather than relying on the header alone.
    """
    return {
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Points": st.column_config.NumberColumn("Points", format="%.0f"),
        "Maximum possible": st.column_config.NumberColumn(
            "Ceiling",
            help="Points this driver would hold after winning everything still available.",
            format="%.0f",
        ),
        "In contention": st.column_config.CheckboxColumn(
            "Alive",
            help="Some sequence of remaining results still leaves this driver champion. "
            "Only false once a rival's current points already beat this driver's ceiling.",
            disabled=True,
        ),
        "Needs no help": st.column_config.CheckboxColumn(
            "Self-sufficient",
            help="Winning every remaining round would be enough on its own, even if the "
            "nearest rival finished second in all of them. False here means still alive, "
            "but now needing a rival to drop points.",
            disabled=True,
        ),
        "Eliminated by": st.column_config.TextColumn("Eliminated by", width="medium"),
    }


def race_calendar_frame(races: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per round of a season's calendar, ready for display.

    Args:
        races: The ``races`` list from a ``SeasonRacesResponse``.

    Returns:
        A frame in calendar order. Rounds not yet run keep their row, with a dash where
        the winner will go -- the calendar is the point, not just the results so far.
    """
    frame = pd.DataFrame(races).sort_values("round")
    frame = frame.assign(
        winner_name=frame["winner_name"].fillna("-"),
        winner_constructor_id=frame["winner_constructor_id"].fillna("-"),
        date=pd.to_datetime(frame["date"], errors="coerce"),
    )
    return frame.rename(
        columns={
            "round": "Rd",
            "name": "Grand Prix",
            "date": "Date",
            "country": "Country",
            "has_sprint": "Sprint",
            "winner_name": "Winner",
            "winner_constructor_id": "Team",
        }
    )[["Rd", "Grand Prix", "Date", "Country", "Sprint", "Winner", "Team"]]


def race_calendar_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`race_calendar_frame`."""
    return {
        "Rd": st.column_config.NumberColumn("Rd", width="small", format="%d"),
        "Grand Prix": st.column_config.TextColumn("Grand Prix", width="large"),
        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        "Country": st.column_config.TextColumn("Country", width="medium"),
        "Sprint": st.column_config.CheckboxColumn("Sprint", disabled=True),
        "Winner": st.column_config.TextColumn("Winner", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
    }


def race_results_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per driver's classification in a single Grand Prix.

    Args:
        results: A ``RaceResultResponse`` list.

    Returns:
        A frame in classified order. Drivers who did not finish show their status ("Retired",
        "+1 Lap") in place of a position, which is how a classification is actually read.
    """
    frame = pd.DataFrame(results).sort_values("position_order")
    return frame.assign(
        position=frame["position"].astype("Int64"),
        grid=frame["grid"].astype("Int64"),
        laps=frame["laps"].astype("Int64"),
    ).rename(
        columns={
            "position": "Pos",
            "driver_name": "Driver",
            "constructor_id": "Team",
            "grid": "Grid",
            "laps": "Laps",
            "status": "Status",
            "points": "Points",
        }
    )[
        ["Pos", "Driver", "Team", "Grid", "Laps", "Status", "Points"]
    ]


def race_results_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`race_results_frame`."""
    return {
        "Pos": st.column_config.NumberColumn(
            "Pos", width="small", format="%d", help="Blank where the driver was not classified."
        ),
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Grid": st.column_config.NumberColumn("Grid", width="small", format="%d"),
        "Laps": st.column_config.NumberColumn("Laps", width="small", format="%d"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Points": st.column_config.NumberColumn("Points", format="%.0f"),
    }


def constructor_standings_frame(standings: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per constructor's championship standing, ready for display.

    Args:
        standings: A ``ConstructorStandingResponse`` list.

    Returns:
        A frame ordered by position with human-readable column names.
    """
    frame = pd.DataFrame(standings).sort_values("position")
    return frame.rename(
        columns={
            "position": "Pos",
            "constructor_name": "Constructor",
            "points": "Points",
            "wins": "Wins",
        }
    )[["Pos", "Constructor", "Points", "Wins"]]


def constructor_standings_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`constructor_standings_frame`."""
    return {
        "Pos": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Constructor": st.column_config.TextColumn("Constructor", width="medium"),
        "Points": st.column_config.NumberColumn("Points", format="%.0f"),
        "Wins": st.column_config.NumberColumn("Wins", format="%d"),
    }


def model_versions_frame(versions: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per trained model version, with metrics flattened into columns.

    Args:
        versions: ``ModelVersionSummary`` dicts from ``/model/versions``.

    Returns:
        A frame with one column per metric key seen across the given versions, newest
        version first.
    """
    if not versions:
        return pd.DataFrame(
            columns=[
                "id",
                "training_run_id",
                "name",
                "algorithm",
                "is_active",
                "is_best",
                "created_at",
            ]
        )

    rows = []
    for version in versions:
        row = {
            "id": version["id"],
            "training_run_id": version["training_run_id"],
            "name": version["name"],
            "algorithm": version["algorithm"],
            "is_active": version["is_active"],
            "is_best": version["is_best"],
            "created_at": version["created_at"],
        }
        row.update(version.get("metrics") or {})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("id", ascending=False)
    # Parsed rather than left as an ISO string so the column sorts chronologically and
    # renders in the reader's own format instead of showing a timezone suffix.
    return frame.assign(created_at=pd.to_datetime(frame["created_at"], format="ISO8601"))


def model_versions_columns(frame: pd.DataFrame) -> dict[str, ColumnConfig]:
    """Formatting for :func:`model_versions_frame`.

    Metric columns vary with what the training run logged, so they are configured from
    the frame itself rather than hard-coded: any column that is not known metadata is a
    metric and gets the shared numeric format.

    Args:
        frame: The frame about to be rendered.

    Returns:
        A column configuration covering both the metadata and the metric columns.
    """
    metadata: dict[str, ColumnConfig] = {
        "id": st.column_config.NumberColumn("ID", width="small", format="%d"),
        "training_run_id": st.column_config.NumberColumn("Run", width="small", format="%d"),
        "name": st.column_config.TextColumn("Model", width="medium"),
        "algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
        "is_active": st.column_config.CheckboxColumn("Live", disabled=True),
        "is_best": st.column_config.CheckboxColumn("Best of run", disabled=True),
        "created_at": st.column_config.DatetimeColumn("Trained", format="YYYY-MM-DD HH:mm"),
    }
    metrics: dict[str, ColumnConfig] = {
        column: st.column_config.NumberColumn(
            column.replace("_", " ").capitalize(), format=_METRIC_FORMAT
        )
        for column in frame.columns
        if column not in metadata
    }
    return {**metadata, **metrics}
