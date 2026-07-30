"""Scoring for walk-forward cross-validation and model selection.

Accuracy is meaningless here (predicting "not champion" for every row scores ~95%), so
everything is built around the actual product question: "how much probability mass did we
put on the eventual champion". :func:`group_log_loss` and :func:`top1_hit_rate` therefore
score the **normalised** probability column -- the one column that is a valid distribution
within a ``(season, round)`` group. :func:`brier_score` and
:func:`expected_calibration_error` instead score the **calibrated, pre-normalisation**
column: they ask "was this driver's individual probability well calibrated", a pointwise
question that normalisation (which redistributes mass across the group) would distort.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "brier_score",
    "compute_metrics",
    "expected_calibration_error",
    "final_round_top1_hit_rate",
    "group_log_loss",
    "top1_hit_rate",
    "top_ranked_mask",
]

_EPS = 1e-15


def group_log_loss(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    group_columns: Sequence[str],
) -> float | None:
    """Mean negative log-probability assigned to the eventual champion, per group.

    Args:
        frame: Rows carrying ``group_columns``, ``target_column`` and
            ``probability_column``.
        probability_column: A per-group probability distribution, typically the
            ``_normalized`` twin of a calibrated probability column.
        target_column: Binary champion label.
        group_columns: Columns identifying a mutually-exclusive group, typically
            ``["season", "round"]``. Used only to assert each group has at most one
            positive row -- a violated assumption would otherwise silently distort the
            mean.

    Returns:
        The mean loss, or ``None`` if no row is labelled positive (nothing to score).

    Raises:
        ValueError: If any group contains more than one positive row.
    """
    positives = frame.filter(pl.col(target_column) == 1)
    if positives.is_empty():
        return None

    duplicated = positives.group_by(list(group_columns)).len().filter(pl.col("len") > 1)
    if duplicated.height:
        msg = f"Groups with more than one positive row: {duplicated.to_dicts()}"
        raise ValueError(msg)

    clipped = positives[probability_column].clip(_EPS, 1.0 - _EPS)
    return float((-clipped.log()).mean())  # type: ignore[arg-type]


def brier_score(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
) -> float | None:
    """Mean squared error between a probability column and the binary label.

    Args:
        frame: Rows carrying ``probability_column`` and ``target_column``.
        probability_column: Predicted probability of ``target_column == 1``.
        target_column: Binary champion label.

    Returns:
        The pooled Brier score, or ``None`` if ``frame`` is empty.
    """
    if frame.is_empty():
        return None
    p = frame[probability_column].to_numpy()
    y = frame[target_column].to_numpy().astype(np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    n_bins: int = 10,
) -> float | None:
    """Weighted mean gap between predicted confidence and observed accuracy, by bin.

    Args:
        frame: Rows carrying ``probability_column`` and ``target_column``.
        probability_column: Predicted probability of ``target_column == 1``.
        target_column: Binary champion label.
        n_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        The expected calibration error, or ``None`` if ``frame`` is empty.
    """
    if frame.is_empty():
        return None
    p = frame[probability_column].to_numpy()
    y = frame[target_column].to_numpy().astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    total = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (count / total) * abs(accuracy - confidence)
    return float(ece)


def top_ranked_mask(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    group_columns: Sequence[str],
) -> pl.Series:
    """Boolean mask selecting each group's single highest-probability row.

    The mechanism behind "who did the model pick as champion": there is no meaningful
    fixed probability threshold against a per-group-normalised distribution, so the
    predicted positive is whichever row ranks first within its group. Used both by
    :func:`top1_hit_rate` and by classification-style metrics (accuracy, precision,
    recall, confusion matrix) elsewhere, which need a discrete predicted label.

    Args:
        frame: Rows carrying ``group_columns`` and ``probability_column``.
        probability_column: Probability used to rank rows within each group.
        group_columns: Columns identifying a mutually-exclusive group.

    Returns:
        A boolean series, ``True`` for the top-ranked row in its group. Ties are broken
        arbitrarily but deterministically.
    """
    rank = frame.select(
        pl.col(probability_column)
        .rank(method="ordinal", descending=True)
        .over(list(group_columns))
        .alias("_rank")
    )["_rank"]
    return rank == 1


def top1_hit_rate(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    group_columns: Sequence[str],
) -> float | None:
    """Fraction of groups where the highest-probability row is the true champion.

    Args:
        frame: Rows carrying ``group_columns``, ``target_column`` and
            ``probability_column``.
        probability_column: Probability used to rank rows within each group.
        target_column: Binary champion label.
        group_columns: Columns identifying a mutually-exclusive group.

    Returns:
        The hit rate, or ``None`` if ``frame`` has no groups.
    """
    if frame.is_empty():
        return None
    top = frame.filter(
        top_ranked_mask(frame, probability_column=probability_column, group_columns=group_columns)
    )
    if top.is_empty():
        return None
    return float(top[target_column].sum()) / top.height


def final_round_top1_hit_rate(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    season_column: str = "season",
    round_column: str = "round",
) -> float | None:
    """Top-1 hit rate restricted to each season's last row present in ``frame``.

    The headline "did we call the right champion" number: whether the model's top-ranked
    driver at the *final* round it saw for a season is the driver who actually won it.

    Args:
        frame: Rows carrying ``season_column``, ``round_column``, ``target_column`` and
            ``probability_column``.
        probability_column: Probability used to rank rows within each season's final round.
        target_column: Binary champion label.
        season_column: Column identifying the season.
        round_column: Column identifying the round within the season.

    Returns:
        The hit rate across seasons present in ``frame``, or ``None`` if empty.
    """
    if frame.is_empty():
        return None
    final_rounds = frame.group_by(season_column).agg(
        pl.col(round_column).max().alias("_final_round")
    )
    final_rows = frame.join(final_rounds, on=season_column, how="inner").filter(
        pl.col(round_column) == pl.col("_final_round")
    )
    return top1_hit_rate(
        final_rows,
        probability_column=probability_column,
        target_column=target_column,
        group_columns=[season_column],
    )


def compute_metrics(
    frame: pl.DataFrame,
    *,
    calibrated_column: str,
    normalized_column: str,
    target_column: str,
    group_columns: Sequence[str],
    calibration_bins: int = 10,
) -> dict[str, float | None]:
    """Compute every selection/reporting metric for one slice of scored rows.

    Called once on the pooled test frame and again on each season-phase subset, so the
    same bundle of numbers is available both overall and stratified.

    Args:
        frame: Scored rows, carrying keys, ``target_column``, ``calibrated_column`` and
            ``normalized_column``.
        calibrated_column: Pointwise calibrated probability column.
        normalized_column: Per-group-normalised probability column.
        target_column: Binary champion label.
        group_columns: Columns identifying a mutually-exclusive group.
        calibration_bins: Number of bins for :func:`expected_calibration_error`.

    Returns:
        A dict with keys ``group_log_loss``, ``brier_score``,
        ``expected_calibration_error`` and ``top1_hit_rate``. Any value is ``None`` when
        ``frame`` has no rows to score it on.
    """
    return {
        "group_log_loss": group_log_loss(
            frame,
            probability_column=normalized_column,
            target_column=target_column,
            group_columns=group_columns,
        ),
        "brier_score": brier_score(
            frame, probability_column=calibrated_column, target_column=target_column
        ),
        "expected_calibration_error": expected_calibration_error(
            frame,
            probability_column=calibrated_column,
            target_column=target_column,
            n_bins=calibration_bins,
        ),
        "top1_hit_rate": top1_hit_rate(
            frame,
            probability_column=normalized_column,
            target_column=target_column,
            group_columns=group_columns,
        ),
    }
