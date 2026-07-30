"""Tests for the dataset schema contract."""

from __future__ import annotations

import polars as pl
import pytest

from f1cp.dataset.schema import validate_dataset

pytestmark = pytest.mark.unit


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020],
            "round": [1, 1, 2, 2],
            "driver_id": ["d1", "d2", "d1", "d2"],
            "rounds_total": [2, 2, 2, 2],
            "current_points": [25.0, 18.0, 43.0, 36.0],
            "champion": [0, 0, 1, 0],
        }
    )


def test_valid_dataset_has_no_violations() -> None:
    violations = validate_dataset(_valid_frame(), target_column="champion", metadata_columns=[])
    assert violations == []


def test_duplicate_key_is_detected() -> None:
    frame = _valid_frame().vstack(_valid_frame()[:1])
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert any(v.check == "duplicate_key" for v in violations)


def test_non_binary_target_is_detected() -> None:
    frame = _valid_frame().with_columns(pl.Series("champion", [0, 0, 2, 0]))
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert any(v.check == "non_binary_target" for v in violations)


def test_multiple_champions_in_one_season_is_detected() -> None:
    frame = _valid_frame().with_columns(pl.Series("champion", [0, 0, 1, 1]))
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert any(v.check == "champion_count" for v in violations)


def test_zero_champions_in_a_completed_season_is_detected() -> None:
    frame = _valid_frame().with_columns(pl.Series("champion", [0, 0, 0, 0]))
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert any(v.check == "champion_count" for v in violations)


def test_in_progress_season_is_not_flagged_for_zero_champions() -> None:
    # `round` (2) has not reached `rounds_total` (4): the season is still being contested,
    # so having no champion yet at the last stored round is expected, not a violation.
    frame = _valid_frame().with_columns(
        pl.Series("rounds_total", [4, 4, 4, 4]), pl.Series("champion", [0, 0, 0, 0])
    )
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert not any(v.check == "champion_count" for v in violations)


def test_non_numeric_feature_column_is_detected() -> None:
    frame = _valid_frame().with_columns(pl.Series("driver_code", ["VER", "HAM", "VER", "HAM"]))
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert any(v.check == "non_numeric_feature" for v in violations)


def test_non_numeric_metadata_column_is_allowed() -> None:
    frame = _valid_frame().with_columns(pl.Series("driver_name", ["A", "B", "C", "D"]))
    violations = validate_dataset(frame, target_column="champion", metadata_columns=["driver_name"])
    assert violations == []


def test_missing_key_column_is_detected() -> None:
    frame = _valid_frame().drop("driver_id")
    violations = validate_dataset(frame, target_column="champion", metadata_columns=[])
    assert [v.check for v in violations] == ["missing_key_column"]
