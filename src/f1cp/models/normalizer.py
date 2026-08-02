"""Per-group probability normalisation.

Winning the championship is mutually exclusive: exactly one driver wins each season. A
model's calibrated probabilities are the output of independent per-row binary classifiers,
though, and have no reason to sum to one within a ``(season, round)`` group. Rescaling them
to do so is what turns "twenty independent probabilities of being *a* champion" into "one
probability distribution over *who* the champion is" -- the question the product actually
answers.

Rescaling is only valid if the group really does contain the answer. Two things guarantee
that, and both are load-bearing:

* every driver in the championship gets a row for every round after their debut, whether or
  not they started that Grand Prix (see
  :func:`f1cp.dataset.builder._apply_filters` for what happened when they did not);
* :func:`f1cp.training.metrics.group_log_loss` refuses to score a group that does not hold
  exactly one champion, rather than skipping it.

:func:`normalize_round_probabilities` is the single entry point training, evaluation and
inference all go through, so the served number and the measured number are produced by the
same three lines of arithmetic. It also applies the one constraint that is exact rather
than learned: a mathematically eliminated driver's probability is set to zero before the
rescale.

A handful of features (``has_sprint``, ``max_points_remaining``, ``round_max_points``,
``rounds_remaining``, ``rounds_total``, ``season_progress``) are constant *within* a group,
so they cannot move any driver's rank relative to another's under this rescaling. That does
not make them dead weight: a constant shifts every row's pre-normalisation logit by the same
amount, and through the sigmoid's non-linearity that shifts how concentrated the resulting
distribution is -- effectively a per-round temperature control, e.g. flatter odds early in a
season with everything still to play for, sharper once ``rounds_remaining`` is small. Do not
remove them for being "unable to change the ranking"; that would degrade calibration for no
gain in the metric it looks like it should help.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "ELIMINATED_COLUMN",
    "GroupProbabilityNormalizer",
    "normalize_round_probabilities",
]

ELIMINATED_COLUMN = "is_mathematically_eliminated"
"""Panel column marking drivers who can no longer catch the championship leader."""


class GroupProbabilityNormalizer:
    """Rescales a probability column to sum to 1 within each group."""

    def __init__(self, group_columns: list[str]) -> None:
        """Initialise the normalizer.

        Args:
            group_columns: Columns identifying a mutually-exclusive group, typically
                ``["season", "round"]``.
        """
        self._group_columns = group_columns

    def normalize(self, frame: pl.DataFrame, probability_column: str) -> pl.DataFrame:
        """Add a normalised twin of ``probability_column``.

        Args:
            frame: A frame with the group columns and a raw/calibrated probability column.
            probability_column: Name of the column to normalise.

        Returns:
            ``frame`` with a ``{probability_column}_normalized`` column added. When a
            group's probabilities sum to zero (a degenerate model output), every member of
            that group is assigned an equal share instead of dividing by zero.
        """
        group_total = pl.col(probability_column).sum().over(self._group_columns)
        group_size = pl.col(probability_column).count().over(self._group_columns)
        return frame.with_columns(
            pl.when(group_total > 0)
            .then(pl.col(probability_column) / group_total)
            .otherwise(1.0 / group_size)
            .alias(f"{probability_column}_normalized")
        )


def normalize_round_probabilities(
    frame: pl.DataFrame,
    *,
    probability_column: str,
    group_columns: list[str],
    enabled: bool = True,
) -> pl.DataFrame:
    """Zero out impossible outcomes, then rescale each round to a distribution.

    The one path from a model's raw output to the number the product shows. Training,
    evaluation and inference all call this rather than assembling the same steps
    themselves, because when they did, they drifted: inference scored drivers training had
    filtered out, so the group being rescaled was not the group that had been measured.

    Elimination is applied as a hard constraint rather than left to the model. Once
    ``gap_to_leader`` exceeds every point still available, a driver's title probability is
    not small, it is zero -- an exact combinatorial fact (see
    :mod:`f1cp.domain.elimination`), already computed as a panel column, and one no amount
    of calibration will reliably reproduce from data. Leaving it to the model meant the
    dashboard could offer title odds on a driver who could not win: on out-of-fold
    predictions, eliminated drivers held ~1.4% of the probability mass of an average round
    and eight of them were shown above 10%. Zeroing first also hands the mass back to
    drivers who *can* still win, which is a small free improvement in group log loss.

    Args:
        frame: Scored rows carrying ``probability_column``, the group columns and -- when
            available -- :data:`ELIMINATED_COLUMN`.
        probability_column: The calibrated probability column to rescale.
        group_columns: Columns identifying a mutually-exclusive group.
        enabled: When false, the normalised column is a copy of the input and no
            constraint is applied, so ``training.normalisation.enabled: false`` still
            yields a frame with the same shape.

    Returns:
        ``frame`` with a ``{probability_column}_normalized`` column added.
    """
    normalized_column = f"{probability_column}_normalized"
    if not enabled:
        return frame.with_columns(pl.col(probability_column).alias(normalized_column))

    constrained = frame
    if ELIMINATED_COLUMN in frame.columns:
        constrained = frame.with_columns(
            pl.when(pl.col(ELIMINATED_COLUMN).fill_null(value=False))
            .then(0.0)
            .otherwise(pl.col(probability_column))
            .alias(probability_column)
        )

    normalized = GroupProbabilityNormalizer(group_columns).normalize(
        constrained, probability_column
    )
    # The constraint belongs to the distribution, not to the pointwise calibrated
    # probability: `brier_score` and `expected_calibration_error` on the calibrated column
    # ask how well the model scored each driver on its own, and must see what it actually
    # produced. Only the normalised column carries the zeroing.
    return normalized.with_columns(frame[probability_column].alias(probability_column))
