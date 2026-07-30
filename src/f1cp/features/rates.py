"""Season-to-date rate features: win, podium, points-finish, pole and finish rates.

Every rate is expanding (cumulative through the previous round), not rolling, since a
career-to-round-R hit rate is a different and complementary signal to the recent-form
windows in :mod:`f1cp.features.form`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["RatesBlock"]

_RATE_SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("is_win", "win_rate"),
    ("is_podium", "podium_rate"),
    ("is_points_finish", "points_finish_rate"),
    ("is_pole", "pole_rate"),
    ("is_top_five", "top_five_rate"),
    ("is_dnf", "dnf_rate"),
)


class RatesBlock(FeatureBlock):
    """Cumulative season-to-date hit rates."""

    name = "rates"
    requires = ("season", "driver_id", "round", *[c for c, _ in _RATE_SOURCE_COLUMNS])

    def build(self, frame: pl.DataFrame, _context: FeatureContext) -> pl.DataFrame:
        """Compute expanding hit rates as of the previous round.

        Args:
            frame: The panel built so far.
            _context: Shared inputs; unused, this block only reads panel columns.

        Returns:
            ``frame`` with rate columns appended, plus ``finish_rate`` (``1 - dnf_rate``).
        """
        group = ["season", "driver_id"]
        rounds_completed = pl.col("round").shift(1).over(group).clip(lower_bound=1)

        out = frame
        for source, target in _RATE_SOURCE_COLUMNS:
            out = out.with_columns(
                (pl.col(source).cast(pl.Int32).shift(1).cum_sum().over(group) / rounds_completed)
                .fill_null(0.0)
                .alias(target)
            )

        return out.with_columns((1.0 - pl.col("dnf_rate")).alias("finish_rate"))
