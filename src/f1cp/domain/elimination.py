"""Championship title contention: who can still mathematically win, in closed form.

Two questions that are easy to confuse, answered separately because only one is
elimination:

* ``still_alive`` -- can this driver still win? Yes, unless a rival *already* holds more
  points than this driver could reach by winning everything left. A rival's total can only
  rise and nothing stops them scoring zero from here, so their current points are the only
  floor that exists.
* ``controls_own_destiny`` -- can they win on their own results alone? Only if winning out
  beats the nearest rival taking second place in every remaining round. Failing this is
  not elimination; it means needing a rival to drop points.

Remaining rounds are independent of one another, so each question has a single optimal
scenario and needs no search over race-by-race outcomes. Ties favour the contender: a
driver who can match the leader is still alive, since countback would then decide the
title and this module does not model it.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1cp.domain.scoring import max_points_per_round, second_best_points_per_round

__all__ = ["TitleContention", "title_contention"]


@dataclass(frozen=True, slots=True)
class TitleContention:
    """Whether one driver can still mathematically win the championship.

    Attributes:
        driver_id: The driver's identifier.
        current_points: Points held right now.
        best_case_points: Points if this driver wins every remaining round (plus every
            sprint win and fastest-lap bonus still available).
        still_alive: Whether any sequence of remaining results leaves this driver on top.
            ``False`` only when a rival's *current* total already exceeds this driver's
            best case -- the point at which no result can change the outcome.
        controls_own_destiny: Whether winning every remaining round is enough on its own,
            even if the nearest rival finishes second in every one of them. A driver can
            be ``still_alive`` without this: they are not eliminated, but they now need a
            rival to drop points somewhere.
        eliminated_by: The ``driver_id`` of a rival whose current points already exceed
            this driver's ceiling, or ``None`` if still alive.
    """

    driver_id: str
    current_points: float
    best_case_points: float
    still_alive: bool
    controls_own_destiny: bool
    eliminated_by: str | None


def title_contention(
    standings: dict[str, float],
    *,
    season: int,
    remaining_rounds: list[bool],
) -> list[TitleContention]:
    """Determine which drivers can still mathematically win the championship.

    Args:
        standings: Current points, keyed by driver id.
        season: Championship year (selects the points system in force).
        remaining_rounds: One entry per round still to be run, ``True`` if it is a sprint
            weekend -- exactly what determines how many points that round can distribute.

    Returns:
        One :class:`TitleContention` per driver, ranked by current points (descending).
    """
    best_case_gain = sum(
        max_points_per_round(season, is_sprint_weekend=is_sprint) for is_sprint in remaining_rounds
    )
    runner_up_gain = sum(
        second_best_points_per_round(season, is_sprint_weekend=is_sprint)
        for is_sprint in remaining_rounds
    )

    results = [
        _contention(driver_id, points, standings, best_case_gain, runner_up_gain)
        for driver_id, points in standings.items()
    ]
    return sorted(results, key=lambda item: item.current_points, reverse=True)


def _contention(
    driver_id: str,
    points: float,
    standings: dict[str, float],
    best_case_gain: float,
    runner_up_gain: float,
) -> TitleContention:
    """Build one driver's outcome against every rival's current and contested totals."""
    ceiling = points + best_case_gain
    rivals = {rival: total for rival, total in standings.items() if rival != driver_id}

    # Elimination: a rival's points can only go up, so their *current* total is the one
    # number this driver's ceiling has to clear. Strictly greater, so a driver who can
    # draw level survives on countback.
    eliminated_by = next(
        (rival for rival, total in rivals.items() if total > ceiling),
        None,
    )

    # Self-sufficiency: the same ceiling, but now every rival is allowed the runner-up
    # spot in every remaining round -- the best they can manage while this driver wins out.
    controls_own_destiny = all(total + runner_up_gain <= ceiling for total in rivals.values())

    return TitleContention(
        driver_id=driver_id,
        current_points=points,
        best_case_points=ceiling,
        still_alive=eliminated_by is None,
        controls_own_destiny=controls_own_destiny,
        eliminated_by=eliminated_by,
    )
