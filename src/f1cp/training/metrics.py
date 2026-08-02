"""Scoring for walk-forward cross-validation and model selection.

Accuracy is meaningless here (predicting "not champion" for every row scores ~95%), so
everything is built around the actual product question: "how much probability mass did we
put on the eventual champion". :func:`group_log_loss` and :func:`top1_hit_rate` therefore
score the **normalised** probability column -- the one column that is a valid distribution
within a ``(season, round)`` group.

Calibration is reported twice, on purpose. :func:`brier_score` and
:func:`expected_calibration_error` on the **calibrated, pre-normalisation** column ask "was
this driver's individual probability well calibrated", a pointwise question normalisation
would distort. The same two on the **normalised** column ask whether the number the product
actually shows is trustworthy, which is the question a reader of the dashboard is really
asking. Reporting only the first was a way of grading work nobody ever sees.

Every headline metric ships next to a baseline computed the same way over the same rows --
"the current championship leader wins the title". A ranking metric without its trivial
baseline is not evidence: at a season's final round that baseline scores exactly 1.000, and
a model that matches it has demonstrated nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BASELINE_RANK_COLUMN",
    "attach_baseline_rank",
    "baseline_top1_hit_rate",
    "brier_score",
    "calibration_metrics",
    "compute_metrics",
    "expected_calibration_error",
    "final_round_rows",
    "final_round_top1_hit_rate",
    "group_log_loss",
    "top1_hit_rate",
    "top_ranked_mask",
]

_EPS = 1e-15

BASELINE_RANK_COLUMN = "baseline_rank"
"""Rank of a driver's championship points within their group, best first.

The reference every ranking metric here is scored against: "the driver leading the
championship after this round is the one who wins it". Attached to the scored frame once,
by :func:`attach_baseline_rank`, so pooled, per-phase and per-season slices all measure the
same baseline over exactly the rows they hold.
"""


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
        The mean loss, or ``None`` if ``frame`` holds no groups at all.

    Raises:
        ValueError: If any group does not contain exactly one positive row. Both directions
            matter and both were silent before. More than one positive would distort the
            mean. *Fewer* than one is worse: it means the group being rescaled to sum to
            one does not contain the answer, so every probability in it is a share of a
            distribution over the wrong set of drivers -- and simply skipping such a group,
            which this function used to do, hides exactly the case where the output is
            meaningless. See :func:`f1cp.dataset.builder._apply_filters` for the defect
            that made this reachable.
    """
    if frame.is_empty():
        return None

    per_group = frame.group_by(list(group_columns)).agg(
        pl.col(target_column).sum().alias("_positives")
    )
    malformed = per_group.filter(pl.col("_positives") != 1)
    if malformed.height:
        msg = (
            f"{malformed.height} group(s) do not contain exactly one positive row, so a "
            f"per-group probability distribution over them is not meaningful: "
            f"{malformed.head(5).to_dicts()}"
        )
        raise ValueError(msg)

    positives = frame.filter(pl.col(target_column) == 1)
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


def attach_baseline_rank(
    frame: pl.DataFrame,
    *,
    points_column: str = "current_points",
    group_columns: Sequence[str],
) -> pl.DataFrame:
    """Add :data:`BASELINE_RANK_COLUMN`: championship rank by points, best first.

    Attached once to the scored frame, before it is sliced by phase or by season, so every
    slice measures the baseline over exactly the rows it holds rather than over some other
    population. Ranked on raw points rather than the provider's ``championship_position``
    because the latter is null for the occasional disqualified driver and ties on
    countback, neither of which a baseline should have an opinion about.

    Args:
        frame: Scored rows carrying ``points_column`` and the group columns.
        points_column: Championship points as of that round.
        group_columns: Columns identifying a mutually-exclusive group.

    Returns:
        ``frame`` with :data:`BASELINE_RANK_COLUMN` added.
    """
    return frame.with_columns(
        pl.col(points_column)
        .rank(method="ordinal", descending=True)
        .over(list(group_columns))
        .alias(BASELINE_RANK_COLUMN)
    )


def baseline_top1_hit_rate(frame: pl.DataFrame, *, target_column: str) -> float | None:
    """Fraction of groups whose championship leader is the eventual champion.

    The number every :func:`top1_hit_rate` in this project has to beat to mean anything.

    Args:
        frame: Rows carrying :data:`BASELINE_RANK_COLUMN` and ``target_column``.
        target_column: Binary champion label.

    Returns:
        The baseline hit rate, or ``None`` if ``frame`` has no rows or no rank column.
    """
    if frame.is_empty() or BASELINE_RANK_COLUMN not in frame.columns:
        return None
    leaders = frame.filter(pl.col(BASELINE_RANK_COLUMN) == 1)
    if leaders.is_empty():
        return None
    return float(leaders[target_column].sum()) / leaders.height


def final_round_rows(
    frame: pl.DataFrame,
    *,
    season_column: str = "season",
    round_column: str = "round",
) -> pl.DataFrame:
    """Each season's last round present in ``frame``.

    Args:
        frame: Rows carrying ``season_column`` and ``round_column``.
        season_column: Column identifying the season.
        round_column: Column identifying the round within the season.

    Returns:
        Only the rows belonging to each season's highest round.
    """
    if frame.is_empty():
        return frame
    final_rounds = frame.group_by(season_column).agg(
        pl.col(round_column).max().alias("_final_round")
    )
    return (
        frame.join(final_rounds, on=season_column, how="inner")
        .filter(pl.col(round_column) == pl.col("_final_round"))
        .drop("_final_round")
    )


def final_round_top1_hit_rate(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    season_column: str = "season",
    round_column: str = "round",
) -> float | None:
    """Top-1 hit rate restricted to each season's last row present in ``frame``.

    Reads as the headline "did we call the right champion" number, and is close to
    worthless on its own: after a season's final round ``max_points_remaining`` is zero, so
    every driver but the points leader is mathematically eliminated and the leader *is* the
    champion by definition. The naive baseline scores exactly 1.000 here. It is kept
    because it is reported beside ``final_round_baseline_top1_hit_rate``, and a metric
    shown next to a baseline it cannot beat teaches more than a metric quietly dropped.

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
    return top1_hit_rate(
        final_round_rows(frame, season_column=season_column, round_column=round_column),
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
        A dict with ``group_log_loss``, ``top1_hit_rate`` and its
        ``baseline_top1_hit_rate``, plus ``brier_score`` and
        ``expected_calibration_error`` on both the calibrated and the normalised column
        (the latter suffixed ``_normalized``). Any value is ``None`` when ``frame`` has no
        rows to score it on.
    """
    return {
        "group_log_loss": group_log_loss(
            frame,
            probability_column=normalized_column,
            target_column=target_column,
            group_columns=group_columns,
        ),
        "top1_hit_rate": top1_hit_rate(
            frame,
            probability_column=normalized_column,
            target_column=target_column,
            group_columns=group_columns,
        ),
        "baseline_top1_hit_rate": baseline_top1_hit_rate(frame, target_column=target_column),
        **calibration_metrics(
            frame,
            probability_column=calibrated_column,
            target_column=target_column,
            n_bins=calibration_bins,
        ),
        **calibration_metrics(
            frame,
            probability_column=normalized_column,
            target_column=target_column,
            n_bins=calibration_bins,
            suffix="_normalized",
        ),
    }


def calibration_metrics(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    target_column: str,
    n_bins: int = 10,
    suffix: str = "",
) -> dict[str, float | None]:
    """Brier score and expected calibration error for one probability column.

    Computed for both the calibrated and the normalised column, because they answer
    different questions and only one of them is what the product shows. See this module's
    docstring.

    Args:
        frame: Rows carrying ``probability_column`` and ``target_column``.
        probability_column: Probability column to score.
        target_column: Binary champion label.
        n_bins: Number of bins for :func:`expected_calibration_error`.
        suffix: Appended to both metric names, to distinguish the two columns.

    Returns:
        ``{"brier_score<suffix>": ..., "expected_calibration_error<suffix>": ...}``.
    """
    return {
        f"brier_score{suffix}": brier_score(
            frame, probability_column=probability_column, target_column=target_column
        ),
        f"expected_calibration_error{suffix}": expected_calibration_error(
            frame,
            probability_column=probability_column,
            target_column=target_column,
            n_bins=n_bins,
        ),
    }
