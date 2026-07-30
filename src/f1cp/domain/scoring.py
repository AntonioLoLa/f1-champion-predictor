"""Formula 1 points systems by era.

The scoring rules changed many times, which makes raw championship points incomparable
across seasons: 25 points for a win in 2026 and 9 points for a win in 1985 describe the
same achievement. Centralising the rules here lets the feature layer produce era-normalised
twins of every points-derived feature from a single source of truth.

Two rules deliberately live outside this module:

* **"Best N results" scoring** (used on and off until 1990) means a driver's championship
  total is not the sum of their race points. Standings are therefore always read from the
  data provider, never recomputed.
* **Shared drives** (1950s) split points between drivers, sometimes into halves.

Sources: FIA sporting regulations; the fastest-lap bonus point existed for 2019-2024 only
and was removed for 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PointsSystem",
    "max_points_per_round",
    "points_for_win",
    "points_system_for",
    "second_best_points_per_round",
    "sprint_points_for_win",
]


@dataclass(frozen=True, slots=True)
class PointsSystem:
    """Scoring rules in force for a range of seasons.

    Attributes:
        first_season: First season the system applied to.
        last_season: Last season it applied to, or ``None`` if still current.
        race_points: Points awarded for P1, P2, ... in a Grand Prix.
        fastest_lap_point: Whether a bonus point is awarded for the fastest lap.
        sprint_points: Points awarded for P1, P2, ... in a sprint race.
    """

    first_season: int
    last_season: int | None
    race_points: tuple[int, ...]
    fastest_lap_point: bool
    sprint_points: tuple[int, ...] = ()

    @property
    def win_points(self) -> int:
        """Points awarded for winning a Grand Prix."""
        return self.race_points[0]

    @property
    def sprint_win_points(self) -> int:
        """Points awarded for winning a sprint race, or 0 if there are no sprints."""
        return self.sprint_points[0] if self.sprint_points else 0

    def covers(self, season: int) -> bool:
        """Whether this system applied in ``season``.

        Args:
            season: Championship year.

        Returns:
            True if the season falls within this system's range.
        """
        if season < self.first_season:
            return False
        return self.last_season is None or season <= self.last_season


_MODERN_RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

# Ordered oldest to newest; ranges are contiguous and non-overlapping.
_POINTS_SYSTEMS: tuple[PointsSystem, ...] = (
    # Top five scored, plus a point for the fastest lap.
    PointsSystem(1950, 1959, (8, 6, 4, 3, 2), fastest_lap_point=True),
    # Fastest-lap point dropped, sixth place starts scoring.
    PointsSystem(1960, 1960, (8, 6, 4, 3, 2, 1), fastest_lap_point=False),
    PointsSystem(1961, 1990, (9, 6, 4, 3, 2, 1), fastest_lap_point=False),
    PointsSystem(1991, 2002, (10, 6, 4, 3, 2, 1), fastest_lap_point=False),
    PointsSystem(2003, 2009, (10, 8, 6, 5, 4, 3, 2, 1), fastest_lap_point=False),
    PointsSystem(2010, 2018, _MODERN_RACE_POINTS, fastest_lap_point=False),
    # Bonus point for the fastest lap (top-10 finishers only) introduced in 2019.
    PointsSystem(2019, 2020, _MODERN_RACE_POINTS, fastest_lap_point=True),
    # Sprint races introduced in 2021 with a 3-2-1 scale for the top three.
    PointsSystem(2021, 2021, _MODERN_RACE_POINTS, True, sprint_points=(3, 2, 1)),
    # Sprint scoring widened to the top eight from 2022.
    PointsSystem(2022, 2024, _MODERN_RACE_POINTS, True, (8, 7, 6, 5, 4, 3, 2, 1)),
    # Fastest-lap bonus point removed for 2025.
    PointsSystem(2025, None, _MODERN_RACE_POINTS, False, (8, 7, 6, 5, 4, 3, 2, 1)),
)


def points_system_for(season: int) -> PointsSystem:
    """Return the points system in force for ``season``.

    Args:
        season: Championship year.

    Returns:
        The applicable :class:`PointsSystem`.

    Raises:
        ValueError: If the season predates the World Championship (1950).
    """
    if season < 1950:
        msg = f"The Formula 1 World Championship began in 1950; got season {season}"
        raise ValueError(msg)

    for system in _POINTS_SYSTEMS:
        if system.covers(season):
            return system

    # Unreachable: the final entry has an open-ended range.
    msg = f"No points system defined for season {season}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def points_for_win(season: int) -> int:
    """Points awarded for a Grand Prix victory in ``season``.

    This is the natural unit for era normalisation: a 30-point gap is worth 1.2 wins in
    2026 but 3.3 wins in 1985.

    Args:
        season: Championship year.

    Returns:
        Points for first place.
    """
    return points_system_for(season).win_points


def sprint_points_for_win(season: int) -> int:
    """Points awarded for a sprint race victory in ``season``.

    Args:
        season: Championship year.

    Returns:
        Points for first place in a sprint, or 0 if the season had no sprints.
    """
    return points_system_for(season).sprint_win_points


def max_points_per_round(season: int, *, is_sprint_weekend: bool = False) -> int:
    """Maximum points a single driver can score over one race weekend.

    Used to compute how many points remain available in a season, which in turn determines
    whether a driver is still mathematically alive in the championship.

    Args:
        season: Championship year.
        is_sprint_weekend: Whether the weekend includes a sprint race.

    Returns:
        The maximum attainable points for the weekend.
    """
    system = points_system_for(season)
    total = system.win_points
    if system.fastest_lap_point:
        total += 1
    if is_sprint_weekend:
        total += system.sprint_win_points
    return total


def second_best_points_per_round(season: int, *, is_sprint_weekend: bool = False) -> int:
    """Maximum points the *second-best* driver can score over one race weekend.

    The tightest possible bound on whether one driver can still catch another: if a driver
    wins every remaining round, every rival is simultaneously capped at second place (plus
    whatever bonus points are still up for grabs) in those same rounds -- never zero. See
    :mod:`f1cp.domain.elimination` for how this is used.

    Args:
        season: Championship year.
        is_sprint_weekend: Whether the weekend includes a sprint race.

    Returns:
        The maximum attainable points for second place over the weekend.
    """
    system = points_system_for(season)
    total = system.race_points[1]
    if system.fastest_lap_point:
        total += 1
    if is_sprint_weekend and len(system.sprint_points) > 1:
        total += system.sprint_points[1]
    return total
