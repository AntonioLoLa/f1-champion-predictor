"""Dataset schema contract.

A hand-rolled contract rather than a validation library dependency: the checks that matter
here -- the key is unique, the label is binary, every season has exactly one champion --
are few, sharp, and cheap to express directly against polars. Failing loudly here is what
turns a silent feature-engineering bug into a build-time error instead of a training-time
mystery.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from f1cp.features.context import KEY_COLUMNS

__all__ = ["SchemaViolation", "validate_dataset"]


@dataclass(frozen=True, slots=True)
class SchemaViolation:
    """One failed contract check.

    Attributes:
        check: Short name of the check that failed.
        detail: Human-readable description of what was found.
    """

    check: str
    detail: str

    def __str__(self) -> str:
        """Render as ``<check>: <detail>``, suitable for a build-time error message."""
        return f"{self.check}: {self.detail}"


def validate_dataset(
    frame: pl.DataFrame,
    *,
    target_column: str,
    metadata_columns: list[str],
) -> list[SchemaViolation]:
    """Run every contract check against a built feature matrix.

    Args:
        frame: The engineered dataset, one row per ``(season, round, driver_id)``.
        target_column: Name of the binary label column.
        metadata_columns: Columns carried for grouping/reporting, excluded from the
            "every feature is numeric" check.

    Returns:
        Every violation found. An empty list means the dataset satisfies the contract.
    """
    violations: list[SchemaViolation] = []
    keys = list(KEY_COLUMNS)

    violations.extend(_check_key_present(frame, keys))
    if any(v.check == "missing_key_column" for v in violations):
        return violations

    violations.extend(_check_key_unique(frame, keys))
    violations.extend(_check_no_null_keys(frame, keys))
    violations.extend(_check_binary_target(frame, target_column))
    violations.extend(_check_one_champion_per_completed_season(frame, target_column))
    violations.extend(_check_feature_dtypes(frame, keys, metadata_columns, target_column))
    return violations


def _check_key_present(frame: pl.DataFrame, keys: list[str]) -> list[SchemaViolation]:
    missing = [key for key in keys if key not in frame.columns]
    if not missing:
        return []
    return [SchemaViolation("missing_key_column", f"key columns absent from dataset: {missing}")]


def _check_key_unique(frame: pl.DataFrame, keys: list[str]) -> list[SchemaViolation]:
    duplicates = frame.height - frame.select(keys).unique().height
    if duplicates == 0:
        return []
    return [
        SchemaViolation("duplicate_key", f"{duplicates} row(s) share a (season, round, driver_id)")
    ]


def _check_no_null_keys(frame: pl.DataFrame, keys: list[str]) -> list[SchemaViolation]:
    violations = []
    for key in keys:
        nulls = frame[key].null_count()
        if nulls:
            violations.append(
                SchemaViolation("null_key", f"{nulls} null value(s) in key column '{key}'")
            )
    return violations


def _check_binary_target(frame: pl.DataFrame, target_column: str) -> list[SchemaViolation]:
    if target_column not in frame.columns:
        return [SchemaViolation("missing_target", f"target column '{target_column}' is absent")]
    values = set(frame[target_column].unique().to_list())
    if not values <= {0, 1}:
        return [
            SchemaViolation(
                "non_binary_target",
                f"target column contains values {values}, expected subset of {{0, 1}}",
            )
        ]
    return []


def _check_one_champion_per_completed_season(
    frame: pl.DataFrame, target_column: str
) -> list[SchemaViolation]:
    """A completed season's final round must have exactly one champion row.

    Only seasons whose full calendar has been ingested are checked (identified via
    ``round == rounds_total`` at the last stored round): the season in progress at build
    time legitimately has zero champions so far, since the label marks the *eventual*
    winner. Without ``rounds_total`` on the frame this check is skipped rather than risking
    a false failure on an in-progress season.
    """
    if target_column not in frame.columns or "rounds_total" not in frame.columns:
        return []

    final_rounds = frame.group_by("season").agg(pl.col("round").max().alias("_final_round"))
    final_rows = (
        frame.join(final_rounds, on="season", how="inner")
        .filter(pl.col("round") == pl.col("_final_round"))
        .filter(pl.col("round") == pl.col("rounds_total"))
    )

    counts = final_rows.group_by("season").agg(pl.col(target_column).sum().alias("_champions"))
    bad = counts.filter(pl.col("_champions") != 1)
    if not bad.height:
        return []

    detail = ", ".join(
        f"{row['season']}={row['_champions']}" for row in bad.sort("season").to_dicts()
    )
    return [
        SchemaViolation(
            "champion_count",
            f"season(s) with champion count != 1 at their final stored round: {detail}",
        )
    ]


def _check_feature_dtypes(
    frame: pl.DataFrame,
    keys: list[str],
    metadata_columns: list[str],
    target_column: str,
) -> list[SchemaViolation]:
    excluded = {*keys, *metadata_columns, target_column}
    non_numeric = [
        column
        for column, dtype in frame.schema.items()
        if column not in excluded and not (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    if not non_numeric:
        return []
    return [
        SchemaViolation(
            "non_numeric_feature",
            f"feature columns must be numeric or boolean, found: {non_numeric}",
        )
    ]
