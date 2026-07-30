"""Tests for closed-form championship title-contention reasoning."""

from __future__ import annotations

import pytest

from f1cp.domain.elimination import title_contention

pytestmark = pytest.mark.unit

# 2026 has no fastest-lap bonus point; a non-sprint round is worth 25/18 (P1/P2), a sprint
# round 33/25 (P1+sprint win / P2+sprint P2).
_SEASON = 2026


def test_leader_with_an_unassailable_lead_eliminates_everyone_else() -> None:
    # Two non-sprint rounds remain: 50 points on offer for the win, 36 for second.
    standings = {"leader": 200.0, "chaser": 100.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False, False])

    by_id = {r.driver_id: r for r in results}
    assert by_id["leader"].still_alive is True
    assert by_id["leader"].eliminated_by is None
    assert by_id["chaser"].still_alive is False
    assert by_id["chaser"].eliminated_by == "leader"


def test_close_title_fight_leaves_both_drivers_alive() -> None:
    # 10 points apart with 50 points still on offer for the win -- a real fight.
    standings = {"leader": 110.0, "chaser": 100.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False, False])

    assert all(r.still_alive for r in results)
    assert all(r.eliminated_by is None for r in results)


def test_a_driver_who_needs_help_is_not_eliminated() -> None:
    """Not self-sufficient and not eliminated are different states, and both can hold.

    Leader has 100, chaser has 80, one non-sprint round remains (25/18 for P1/P2). The
    chaser's ceiling is 105, which clears the leader's current 100 -- so there *is* a
    sequence of results where the chaser takes the title (they win, the leader scores
    nothing) and they are not eliminated. What they cannot do is guarantee it: if the
    leader takes second in that same round they finish on 118, so the chaser no longer
    controls their own destiny.
    """
    standings = {"leader": 100.0, "chaser": 80.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False])

    by_id = {r.driver_id: r for r in results}
    assert by_id["chaser"].still_alive is True
    assert by_id["chaser"].controls_own_destiny is False
    assert by_id["chaser"].best_case_points == pytest.approx(105.0)


def test_a_rival_scoring_nothing_is_always_a_permitted_scenario() -> None:
    """The heart of the fix: rivals have a floor of zero, not of second place.

    A rival's points can only rise, so their *current* total is the only bound elimination
    can rest on. Assuming instead that a rival banks the runner-up spot every round is an
    assumption about how well they will do -- an upper bound on the rival, which cannot
    eliminate anyone.
    """
    # 11 rounds left, so 275 points are on offer: enough for the chaser to overhaul a
    # 50-point deficit outright if the leader stops scoring.
    standings = {"leader": 219.0, "chaser": 138.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False] * 11)

    by_id = {r.driver_id: r for r in results}
    assert by_id["chaser"].still_alive is True
    assert by_id["chaser"].eliminated_by is None
    # ...but the leader taking second every round would put them on 417, above the
    # chaser's 413 ceiling, so the chaser cannot do it on their own results alone.
    assert by_id["chaser"].controls_own_destiny is False


def test_a_big_enough_lead_makes_a_driver_self_sufficient() -> None:
    standings = {"leader": 300.0, "chaser": 100.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False])

    by_id = {r.driver_id: r for r in results}
    assert by_id["leader"].controls_own_destiny is True
    assert by_id["chaser"].controls_own_destiny is False


def test_no_rounds_remaining_leaves_only_the_sole_leader_alive() -> None:
    standings = {"champion": 300.0, "runner_up": 280.0, "third": 250.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[])

    by_id = {r.driver_id: r for r in results}
    assert by_id["champion"].still_alive is True
    assert by_id["runner_up"].still_alive is False
    assert by_id["third"].still_alive is False


def test_an_exact_final_tie_eliminates_neither() -> None:
    """A strict, not >=, comparison: an exact tie is not treated as elimination."""
    standings = {"a": 300.0, "b": 300.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[])

    assert all(r.still_alive for r in results)


def test_results_are_ranked_by_current_points_descending() -> None:
    standings = {"midfield": 50.0, "leader": 200.0, "backmarker": 10.0}
    results = title_contention(standings, season=_SEASON, remaining_rounds=[False])

    assert [r.driver_id for r in results] == ["leader", "midfield", "backmarker"]


def test_sprint_rounds_are_worth_more_than_conventional_rounds() -> None:
    standings = {"leader": 200.0, "chaser": 100.0}
    conventional = title_contention(standings, season=_SEASON, remaining_rounds=[False])
    sprint = title_contention(standings, season=_SEASON, remaining_rounds=[True])

    conventional_ceiling = next(r for r in conventional if r.driver_id == "chaser").best_case_points
    sprint_ceiling = next(r for r in sprint if r.driver_id == "chaser").best_case_points
    assert sprint_ceiling > conventional_ceiling
