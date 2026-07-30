"""Per-group probability normalisation.

Winning the championship is mutually exclusive: exactly one driver wins each season. A
model's calibrated probabilities are the output of independent per-row binary classifiers,
though, and have no reason to sum to one within a ``(season, round)`` group. Rescaling them
to do so is what turns "twenty independent probabilities of being *a* champion" into "one
probability distribution over *who* the champion is" -- the question the product actually
answers.
"""

from __future__ import annotations

import polars as pl

__all__ = ["GroupProbabilityNormalizer"]


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
