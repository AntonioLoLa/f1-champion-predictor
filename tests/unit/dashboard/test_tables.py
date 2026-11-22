"""The frames behind the dashboard's tables: what a reader sees in each column."""

from __future__ import annotations

from typing import Any

import pytest

from f1cp.dashboard.components.tables import (
    contention_frame,
    prediction_rows_frame,
    race_calendar_frame,
)

pytestmark = pytest.mark.unit


def _driver(driver_id: str, rank: int, probability: float, position: int) -> dict[str, Any]:
    return {
        "driver_id": driver_id,
        "driver_name": driver_id.title(),
        "constructor_id": "red-bull",
        "constructor_name": "Red Bull",
        "points": 100.0 - position,
        "standing_position": position,
        "probability": probability,
        "probability_calibrated": probability,
        "probability_raw": probability,
        "rank": rank,
    }


def test_predictions_show_team_names_and_standings_without_model_diagnostics() -> None:
    frame = prediction_rows_frame([_driver("b", 2, 0.0, 2), _driver("a", 1, 1.0, 1)])

    assert frame["Driver"].tolist() == ["A", "B"]
    assert set(frame["Team"]) == {"Red Bull"}
    assert frame["Standing"].tolist() == [1, 2]
    assert "Calibrated" not in frame.columns
    assert "Raw" not in frame.columns


def test_model_diagnostics_are_available_on_request() -> None:
    frame = prediction_rows_frame([_driver("a", 1, 1.0, 1)], diagnostics=True)

    assert {"Calibrated", "Raw"} <= set(frame.columns)


def test_contention_shows_each_drivers_deficit_to_the_leader() -> None:
    contenders = [
        {
            "driver_name": name,
            "current_points": points,
            "best_case_points": points + 75,
            "still_alive": alive,
            "controls_own_destiny": alive,
            "eliminated_by_name": None if alive else "Kimi Antonelli",
        }
        for name, points, alive in [
            ("Max Verstappen", 279.0, False),
            ("Kimi Antonelli", 390.0, True),
        ]
    ]

    frame = contention_frame(contenders)

    assert frame["Driver"].tolist() == ["Kimi Antonelli", "Max Verstappen"]
    assert frame["Behind leader"].tolist() == [0.0, 111.0]


def test_the_calendar_names_each_winners_team() -> None:
    races = [
        {
            "round": 1,
            "name": "Australian Grand Prix",
            "date": "2026-03-08",
            "country": "Australia",
            "has_sprint": False,
            "winner_name": "George Russell",
            "winner_constructor_id": "mercedes",
            "winner_constructor_name": "Mercedes",
        },
        {
            "round": 2,
            "name": "Chinese Grand Prix",
            "date": "2026-03-15",
            "country": "China",
            "has_sprint": True,
            "winner_name": None,
            "winner_constructor_id": None,
            "winner_constructor_name": None,
        },
    ]

    assert race_calendar_frame(races)["Team"].tolist() == ["Mercedes", "-"]
