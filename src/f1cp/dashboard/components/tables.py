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

from f1cp.dashboard.components.labels import (
    metric_help,
    metric_label,
    metric_order,
    model_label,
    team_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from streamlit.elements.lib.column_types import ColumnConfig

__all__ = [
    "constructor_standings_columns",
    "constructor_standings_frame",
    "contention_columns",
    "contention_frame",
    "data_table",
    "model_versions_columns",
    "model_versions_frame",
    "pit_stops_columns",
    "pit_stops_frame",
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
    columns: Mapping[str, ColumnConfig | None] | None = None,
    *,
    height: int | None = None,
) -> None:
    """Render a frame as the app's one table style.

    Args:
        frame: The data to show.
        columns: Per-column formatting, from the matching ``*_columns`` builder. A
            ``None`` value hides that column.
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


def _with_team(rows: list[dict[str, Any]], *, prefix: str = "") -> pd.DataFrame:
    """``rows`` as a frame with a ``team`` column holding each row's team display name.

    Resolved from the API rows before they become a frame, so every table names the team
    the same way -- see :func:`~f1cp.dashboard.components.labels.team_name`.
    """
    return pd.DataFrame([{**row, "team": team_name(row, prefix=prefix)} for row in rows])


def prediction_rows_frame(
    drivers: list[dict[str, Any]], *, diagnostics: bool = False
) -> pd.DataFrame:
    """One row per driver's ranked prediction, ready for display.

    Args:
        drivers: The ``drivers`` list from a ``RoundPredictionsResponse``.
        diagnostics: Also include the model's probability before calibration and before
            rescaling across the field. Those are for checking the model, not for reading
            the title race, so the default table leaves them out.

    Returns:
        A frame ordered by rank with human-readable column names.
    """
    frame = _with_team(drivers).sort_values("rank")
    frame = frame.assign(
        standing_position=frame["standing_position"].astype("Int64"),
    ).rename(
        columns={
            "rank": "Rank",
            "driver_name": "Driver",
            "team": "Team",
            "standing_position": "Standing",
            "points": "Points",
            "probability": "Title chance",
            "probability_calibrated": "Calibrated",
            "probability_raw": "Raw",
        }
    )
    columns = ["Rank", "Driver", "Team", "Standing", "Points", "Title chance"]
    if diagnostics:
        columns += ["Calibrated", "Raw"]
    return frame[columns]


def prediction_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`prediction_rows_frame`.

    The headline probability renders as a bar rather than a number: it is the column the
    reader scans, and a bar is read without arithmetic. The two diagnostic probabilities
    behind it stay as plain percentages -- they are for checking, not scanning.
    """
    return {
        "Rank": st.column_config.NumberColumn(
            "#",
            width="small",
            format="%d",
            help="Order by title chance. Drivers with equal odds are listed in "
            "championship order.",
        ),
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Team": st.column_config.TextColumn("Team", width="medium"),
        "Standing": st.column_config.NumberColumn(
            "Pos",
            width="small",
            format="P%d",
            help="Position in the drivers' championship after this round.",
        ),
        "Points": st.column_config.NumberColumn("Points", format="%.0f"),
        "Title chance": st.column_config.ProgressColumn(
            "Title chance",
            help="Probability of winning the championship, rescaled to sum to 100% "
            "across the field.",
            format="percent",
            min_value=0.0,
            max_value=1.0,
        ),
        "Calibrated": st.column_config.NumberColumn(
            "Before rescaling",
            help="The model's probability for this driver alone, before the field is "
            "rescaled to share 100% between them.",
            format="percent",
        ),
        "Raw": st.column_config.NumberColumn(
            "Uncalibrated",
            help="The model's direct output, before it is adjusted to match how often "
            "similar odds came true in past seasons.",
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
    frame = _with_team(standings).sort_values("position")
    return frame.rename(
        columns={
            "position": "Pos",
            "driver_name": "Driver",
            "team": "Team",
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
    frame = pd.DataFrame(contenders).sort_values(
        ["current_points", "driver_name"], ascending=[False, True]
    )
    frame = frame.assign(
        behind=frame["current_points"].max() - frame["current_points"],
        eliminated_by=frame.apply(
            lambda row: "-" if row["still_alive"] else (row["eliminated_by_name"] or "-"),
            axis=1,
        ),
    )
    return frame.rename(
        columns={
            "driver_name": "Driver",
            "current_points": "Points",
            "behind": "Behind leader",
            "best_case_points": "Maximum possible",
            "still_alive": "In contention",
            "controls_own_destiny": "Needs no help",
            "eliminated_by": "Eliminated by",
        }
    )[
        [
            "Driver",
            "Points",
            "Behind leader",
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
        "Behind leader": st.column_config.NumberColumn(
            "Behind leader", help="Points behind the championship leader.", format="%.0f"
        ),
        "Maximum possible": st.column_config.NumberColumn(
            "Max possible",
            help="Points this driver would hold after winning every race and sprint still "
            "to come.",
            format="%.0f",
        ),
        "In contention": st.column_config.CheckboxColumn(
            "Can still win",
            help="Some sequence of remaining results still leaves this driver champion. "
            "Only false once a rival already has more points than this driver can still reach.",
            disabled=True,
        ),
        "Needs no help": st.column_config.CheckboxColumn(
            "In own hands",
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
    frame = _with_team(races, prefix="winner_").sort_values("round")
    frame = frame.assign(
        winner_name=frame["winner_name"].fillna("-"),
        team=frame["team"].fillna("-"),
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
            "team": "Team",
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
    frame = _with_team(results).sort_values("position_order")
    return frame.assign(
        position=frame["position"].astype("Int64"),
        grid=frame["grid"].astype("Int64"),
        laps=frame["laps"].astype("Int64"),
    ).rename(
        columns={
            "position": "Pos",
            "driver_name": "Driver",
            "team": "Team",
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


def pit_stops_frame(pit_stops: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per pit stop, driver then stop number.

    Args:
        pit_stops: A ``PitStopResponse`` list.

    Returns:
        A frame with the stop duration in seconds, which is how a pit stop is actually
        talked about, rather than the milliseconds the source data stores it in.
    """
    frame = pd.DataFrame(pit_stops)
    return frame.assign(
        lap=frame["lap"].astype("Int64"),
        duration_s=frame["duration_ms"] / 1000,
    ).rename(
        columns={
            "driver_name": "Driver",
            "stop": "Stop",
            "lap": "Lap",
            "duration_s": "Duration (s)",
        }
    )[
        ["Driver", "Stop", "Lap", "Duration (s)"]
    ]


def pit_stops_columns() -> dict[str, ColumnConfig]:
    """Formatting for :func:`pit_stops_frame`."""
    return {
        "Driver": st.column_config.TextColumn("Driver", width="medium"),
        "Stop": st.column_config.NumberColumn("Stop", width="small", format="%d"),
        "Lap": st.column_config.NumberColumn("Lap", width="small", format="%d"),
        "Duration (s)": st.column_config.NumberColumn("Duration (s)", format="%.3f"),
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
                "model",
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
            "model": model_label(version["name"]),
        }
        row.update(version.get("metrics") or {})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("id", ascending=False)
    # Columns in reading order: which model, whether it serves, then its scores with the
    # one it was selected on first; bookkeeping last.
    leading = ["model", "is_active", "is_best"]
    trailing = ["created_at", "id", "training_run_id", "name", "algorithm"]
    metrics = metric_order([c for c in frame.columns if c not in {*leading, *trailing}])
    frame = frame[[*leading, *metrics, *trailing]]
    # Parsed rather than left as an ISO string so the column sorts chronologically and
    # renders in the reader's own format instead of showing a timezone suffix.
    return frame.assign(created_at=pd.to_datetime(frame["created_at"], format="ISO8601"))


def model_versions_columns(
    frame: pd.DataFrame, *, show_ids: bool = False
) -> dict[str, ColumnConfig | None]:
    """Formatting for :func:`model_versions_frame`.

    Metric columns vary with what the training run logged, so they are configured from
    the frame itself rather than hard-coded: any column that is not known metadata is a
    metric and gets the shared numeric format, with its plain-language name and an
    explanation of which direction is better.

    Args:
        frame: The frame about to be rendered.
        show_ids: Show the version and training-run ids. Useful in the full training
            history, where they are what tells otherwise identical rows apart; noise
            everywhere else.

    Returns:
        A column configuration covering both the metadata and the metric columns. A
        ``None`` entry hides that column.
    """
    ids: dict[str, ColumnConfig | None] = {
        "id": st.column_config.NumberColumn("Version", width="small", format="%d"),
        "training_run_id": st.column_config.NumberColumn("Run", width="small", format="%d"),
    }
    metadata: dict[str, ColumnConfig | None] = {
        **(ids if show_ids else dict.fromkeys(ids)),
        # The internal key; the readable `model` column below is what is shown.
        "name": None,
        "algorithm": None,
        "model": st.column_config.TextColumn("Model", width="medium"),
        "is_active": st.column_config.CheckboxColumn("Live", disabled=True),
        "is_best": st.column_config.CheckboxColumn("Best of run", disabled=True),
        "created_at": st.column_config.DatetimeColumn("Trained", format="YYYY-MM-DD HH:mm"),
    }
    metrics: dict[str, ColumnConfig | None] = {
        column: st.column_config.NumberColumn(
            metric_label(column), help=metric_help(column), format=_METRIC_FORMAT
        )
        for column in frame.columns
        if column not in metadata
    }
    return {**metadata, **metrics}
