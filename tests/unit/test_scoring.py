"""Tests for the historical points systems."""

from __future__ import annotations

import pytest

from f1cp.domain import (
    max_points_per_round,
    points_for_win,
    points_system_for,
    second_best_points_per_round,
    sprint_points_for_win,
)
from f1cp.domain.scoring import _POINTS_SYSTEMS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (1950, 8),
        (1960, 8),
        (1961, 9),
        (1990, 9),
        (1991, 10),
        (2002, 10),
        (2003, 10),
        (2009, 10),
        (2010, 25),
        (2026, 25),
    ],
)
def test_points_for_win_matches_the_era(season: int, expected: int) -> None:
    assert points_for_win(season) == expected


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (2018, False),  # bonus point did not exist yet
        (2019, True),  # introduced
        (2024, True),  # final season with it
        (2025, False),  # removed by the FIA for 2025
        (2026, False),
    ],
)
def test_fastest_lap_bonus_point_window(season: int, *, expected: bool) -> None:
    assert points_system_for(season).fastest_lap_point is expected


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        (2020, 0),  # no sprints
        (2021, 3),  # 3-2-1 for the top three
        (2022, 8),  # widened to the top eight
        (2026, 8),
    ],
)
def test_sprint_win_points(season: int, expected: int) -> None:
    assert sprint_points_for_win(season) == expected


@pytest.mark.parametrize(
    ("season", "is_sprint", "expected"),
    [
        (2018, False, 25),  # win only
        (2019, False, 26),  # win + fastest lap
        (2021, True, 29),  # win + fastest lap + sprint win (3)
        (2023, True, 34),  # win + fastest lap + sprint win (8)
        (2026, True, 33),  # win + sprint win, no fastest-lap point
        (2026, False, 25),
    ],
)
def test_max_points_per_round(season: int, expected: int, *, is_sprint: bool) -> None:
    assert max_points_per_round(season, is_sprint_weekend=is_sprint) == expected


@pytest.mark.parametrize(
    ("season", "is_sprint", "expected"),
    [
        (2018, False, 18),  # P2 only
        (2019, False, 19),  # P2 + fastest lap
        (2021, True, 21),  # P2 (18) + fastest lap (1) + sprint P2 (2)
        (2026, True, 25),  # P2 (18) + sprint P2 (7), no fastest-lap point
        (2026, False, 18),
    ],
)
def test_second_best_points_per_round(season: int, expected: int, *, is_sprint: bool) -> None:
    assert second_best_points_per_round(season, is_sprint_weekend=is_sprint) == expected


def test_season_before_the_championship_is_rejected() -> None:
    with pytest.raises(ValueError, match="began in 1950"):
        points_system_for(1949)


def test_points_systems_form_a_contiguous_non_overlapping_timeline() -> None:
    """A gap or an overlap would silently return the wrong era for some season."""
    for earlier, later in zip(_POINTS_SYSTEMS, _POINTS_SYSTEMS[1:], strict=False):
        assert earlier.last_season is not None
        assert later.first_season == earlier.last_season + 1

    assert _POINTS_SYSTEMS[0].first_season == 1950
    assert _POINTS_SYSTEMS[-1].last_season is None


def test_race_points_are_strictly_decreasing() -> None:
    for system in _POINTS_SYSTEMS:
        points = system.race_points
        assert all(a > b for a, b in zip(points, points[1:], strict=False)), system
