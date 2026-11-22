"""Words for the reader: how internal names and raw numbers are presented in the UI.

The API speaks in identifiers -- ``red-bull``, ``logistic_regression``,
``gap_to_leader_over_remaining``, ``group_log_loss`` -- because those are stable keys for
code. None of them is something a Formula 1 fan should have to decode, so every page goes
through this module to turn a key into a label, and never prints one directly.

Each lookup falls back to a tidied version of the key rather than raising: a feature or
metric added after this module was last edited should read as slightly plain, not break a
page. ``tests/integration/test_dashboard.py`` guards the feature labels against that drift
by checking every column the current feature configuration produces has a curated label.

Like :mod:`~f1cp.dashboard.components.theme`, nothing here fetches or transforms data.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FEATURE_LABELS",
    "feature_label",
    "metric_help",
    "metric_label",
    "metric_order",
    "model_label",
    "percent",
    "team_name",
]

# The smallest probability `percent` shows as a number. Below it, one decimal place would
# round to "0.0%" -- which reads as "eliminated" for a driver who is not.
_SMALLEST_SHOWN = 0.0005

_MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "Neural network",
}

# (label, explanation). Every explanation says which direction is better, because the
# leaderboard mixes losses (lower wins) with hit rates (higher wins).
_METRICS: dict[str, tuple[str, str]] = {
    "group_log_loss": (
        "Log loss",
        "How surprised the model was, on average, by who actually became champion. "
        "The number models are selected on. Lower is better.",
    ),
    "top1_hit_rate": (
        "Favourite became champion",
        "Share of rounds where the driver the model rated highest went on to win the "
        "title. Higher is better.",
    ),
    "baseline_top1_hit_rate": (
        "Points leader became champion",
        "The same share for the naive guess 'whoever leads the standings wins'. The "
        "model is only useful where it beats this.",
    ),
    "final_round_top1_hit_rate": (
        "Favourite became champion (season finale)",
        "As 'Favourite became champion', counting only each season's last round.",
    ),
    "final_round_baseline_top1_hit_rate": (
        "Points leader became champion (season finale)",
        "As 'Points leader became champion', counting only each season's last round.",
    ),
    "brier_score": (
        "Brier score (per driver)",
        "Squared error of each driver's own probability, before the field is rescaled "
        "to 100%. Lower is better.",
    ),
    "brier_score_normalized": (
        "Brier score",
        "Squared error of the title odds as shown in this app. Lower is better.",
    ),
    "expected_calibration_error": (
        "Calibration error (per driver)",
        "How far each driver's own probability sits from how often such probabilities "
        "came true, before rescaling. Lower is better.",
    ),
    "expected_calibration_error_normalized": (
        "Calibration error",
        "How far the title odds shown in this app sit from how often they came true: "
        "a well calibrated 30% wins about 30% of the time. Lower is better.",
    ),
}

FEATURE_LABELS: dict[str, str] = {
    # Championship standing
    "current_points": "Championship points",
    "current_points_norm": "Championship points (in race wins)",
    "championship_position": "Championship position",
    "gap_to_leader": "Points behind the leader",
    "gap_to_leader_norm": "Points behind the leader (in race wins)",
    "gap_to_second": "Points behind second place",
    "gap_to_second_norm": "Points behind second place (in race wins)",
    "gap_to_leader_over_remaining": "Deficit to the leader vs points still available",
    "is_leader": "Leads the championship",
    "points_share": "Share of all points scored this season",
    "season_wins": "Wins this season",
    "is_mathematically_eliminated": "Mathematically eliminated",
    # Season context
    "rounds_total": "Rounds on the calendar",
    "rounds_remaining": "Rounds still to run",
    "season_progress": "Share of the season completed",
    "max_points_remaining": "Points still available",
    "round_max_points": "Points available this weekend",
    "has_sprint": "Sprint weekend",
    # This weekend's result
    "finish_position": "Finishing position",
    "grid": "Starting grid position",
    "quali_position": "Qualifying position",
    "positions_gained": "Places gained in the race",
    "sprint_position": "Sprint finishing position",
    "race_points": "Grand Prix points",
    "sprint_points": "Sprint points",
    "round_points": "Points this weekend",
    "entered": "Took part this weekend",
    "is_win": "Won the race",
    "is_podium": "Finished on the podium",
    "is_top_five": "Finished in the top five",
    "is_points_finish": "Finished in the points",
    "is_pole": "Started from pole",
    "is_dnf": "Did not finish",
    # Recent form
    "last_3_points": "Points over the last 3 races",
    "last_5_points": "Points over the last 5 races",
    "last_3_avg_finish": "Average finish, last 3 races",
    "last_5_avg_finish": "Average finish, last 5 races",
    "last_3_avg_grid": "Average grid slot, last 3 races",
    "season_points_rate": "Points per race this season",
    "momentum": "Recent points rate vs season average",
    "trend": "Finishing-position trend",
    "consistency": "Spread of recent finishing positions",
    "rounds_since_win": "Races since last win",
    "rounds_since_podium": "Races since last podium",
    # Season rates
    "win_rate": "Win rate this season",
    "podium_rate": "Podium rate this season",
    "top_five_rate": "Top-five rate this season",
    "points_finish_rate": "Points-finish rate this season",
    "pole_rate": "Pole rate this season",
    "dnf_rate": "Retirement rate this season",
    # Qualifying
    "season_avg_grid": "Average grid slot this season",
    "season_avg_quali": "Average qualifying position this season",
    "season_best_grid": "Best grid slot this season",
    "season_avg_positions_gained": "Average places gained per race this season",
    # Team
    "team_points": "Team points",
    "team_points_last_3": "Team points over the last 3 races",
    "team_points_share": "Driver's share of the team's points",
    "team_position": "Team's championship position",
    "team_wins": "Team wins this season",
    "team_dnf_rate": "Team retirement rate this season",
    "teammate_beat_rate": "How often they beat their teammate",
    "teammate_points_delta": "Points ahead of their teammate",
    # Career
    "age": "Age",
    "career_starts": "Career starts",
    "career_wins": "Career wins",
    "career_win_rate": "Career win rate",
    "experience_seasons": "Seasons in Formula 1",
    "prior_titles": "Previous titles",
    "is_defending_champion": "Defending champion",
    # Remaining calendar
    "avg_finish_at_remaining_circuits": "Past average finish at the circuits still to come",
    "remaining_circuits_visited": "Circuits still to come they have raced at",
    "remaining_home_races": "Home races still to come",
}


def _tidy(key: str) -> str:
    """``some_internal_key`` as ``Some internal key``: the fallback for an unknown name."""
    text = key.replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:]


def model_label(name: str | None) -> str:
    """A model's display name, e.g. ``"Logistic regression"`` for ``logistic_regression``.

    Args:
        name: The model version's name or algorithm key.

    Returns:
        A readable name; ``"No model"`` when there is none.
    """
    if not name:
        return "No model"
    return _MODEL_LABELS.get(name, _tidy(name))


def metric_label(key: str) -> str:
    """A metric's column header, e.g. ``"Log loss"`` for ``group_log_loss``."""
    known = _METRICS.get(key)
    return known[0] if known else _tidy(key)


def metric_help(key: str) -> str | None:
    """What a metric measures and which direction is better, or ``None`` if unknown."""
    known = _METRICS.get(key)
    return known[1] if known else None


def metric_order(keys: list[str]) -> list[str]:
    """``keys`` in reading order: the selection metric first, then the curated list.

    Metrics this module does not know about follow, alphabetically.
    """
    known = [key for key in _METRICS if key in keys]
    return known + sorted(key for key in keys if key not in _METRICS)


def feature_label(key: str) -> str:
    """A model input in plain words, e.g. ``"Points behind the leader"``."""
    return FEATURE_LABELS.get(key, _tidy(key))


def team_name(row: dict[str, Any], *, prefix: str = "") -> str | None:
    """A team's display name from an API row, falling back to a tidied id.

    Args:
        row: A response row carrying ``{prefix}constructor_name`` and
            ``{prefix}constructor_id``.
        prefix: Field prefix, e.g. ``"winner_"`` for a calendar row.

    Returns:
        The team name, or ``None`` when the row names no team at all.
    """
    name = row.get(f"{prefix}constructor_name")
    if name:
        return str(name)
    team_id = row.get(f"{prefix}constructor_id")
    return _tidy(str(team_id)) if team_id else None


def percent(probability: float) -> str:
    """A probability as the reader should see it.

    Exactly zero shows as ``"0%"`` -- that driver cannot win. Anything above zero but too
    small for one decimal place shows as ``"<0.1%"`` rather than rounding to a zero that
    would claim the same thing, and the mirror image holds at the top: only a certainty
    reads ``"100%"``.

    Args:
        probability: A 0-1 probability.

    Returns:
        The formatted percentage.
    """
    if probability <= 0:
        return "0%"
    if probability >= 1:
        return "100%"
    if probability < _SMALLEST_SHOWN:
        return "<0.1%"
    if probability > 1 - _SMALLEST_SHOWN:
        return ">99.9%"
    return f"{probability:.1%}"
