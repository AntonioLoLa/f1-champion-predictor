"""How internal names and raw probabilities are presented to the reader."""

from __future__ import annotations

import pytest

from f1cp.dashboard.components.labels import (
    feature_label,
    metric_help,
    metric_label,
    metric_order,
    model_label,
    percent,
    team_name,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("probability", "shown"),
    [
        (0.0, "0%"),
        (0.0001, "<0.1%"),
        (0.0005, "0.1%"),
        (0.384, "38.4%"),
        (0.9999, ">99.9%"),
        (1.0, "100%"),
    ],
)
def test_only_certainty_reads_as_zero_or_one_hundred(probability: float, shown: str) -> None:
    # A driver who can still win must never read "0.0%", which says they cannot.
    assert percent(probability) == shown


def test_a_team_is_named_not_slugged() -> None:
    assert team_name({"constructor_id": "red-bull", "constructor_name": "Red Bull"}) == "Red Bull"
    # No name in the response: a tidied id is still better than the raw slug.
    assert team_name({"constructor_id": "red-bull", "constructor_name": None}) == "Red bull"
    assert team_name({"constructor_id": None}) is None


def test_a_calendar_row_names_the_winners_team() -> None:
    row = {"winner_constructor_id": "mercedes", "winner_constructor_name": "Mercedes"}
    assert team_name(row, prefix="winner_") == "Mercedes"


def test_model_and_metric_keys_read_as_words() -> None:
    assert model_label("logistic_regression") == "Logistic regression"
    assert model_label("some_new_model") == "Some new model"
    assert metric_label("group_log_loss") == "Log loss"
    assert "Lower is better" in (metric_help("group_log_loss") or "")
    assert metric_label("a_metric_added_later") == "A metric added later"
    assert metric_help("a_metric_added_later") is None


def test_features_read_as_words() -> None:
    assert feature_label("gap_to_leader") == "Points behind the leader"
    assert feature_label("a_feature_added_later") == "A feature added later"


def test_the_selection_metric_leads_and_unknown_metrics_trail() -> None:
    assert metric_order(["zz_new", "top1_hit_rate", "group_log_loss", "aa_new"]) == [
        "group_log_loss",
        "top1_hit_rate",
        "aa_new",
        "zz_new",
    ]
