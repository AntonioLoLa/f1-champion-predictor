"""Season-to-date rate features: win, podium, points-finish, pole, top-five and DNF rates.

Every rate is expanding (cumulative through the previous round), not rolling, since a
season-to-round-R hit rate is a different and complementary signal to the recent-form
windows in :mod:`f1cp.features.form`. All of them divide by rounds the driver actually
*started*, which is what makes them comparable between a driver who has been there since
round one and one who joined at round fourteen.

There is deliberately no ``finish_rate``: it was ``1 - dnf_rate``, carried no information
of its own, and the perfect anti-correlation between the two split their attributed
importance in half and drove permutation importance for both to approximately zero.
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
    requires = ("season", "driver_id", "round", "entered", *[c for c, _ in _RATE_SOURCE_COLUMNS])

    def build(self, frame: pl.DataFrame, _context: FeatureContext) -> pl.DataFrame:
        """Compute expanding hit rates as of the previous round.

        Args:
            frame: The panel built so far.
            _context: Shared inputs; unused, this block only reads panel columns.

        Returns:
            ``frame`` with rate columns appended.
        """
        group = ["season", "driver_id"]
        # Rounds this driver has actually started, not rounds the calendar has run. The
        # two differ for anyone who joins mid-season, and dividing by the calendar made a
        # debutant look uniformly excellent: a driver who entered at round 14 and retired
        # on debut was scored with a DNF rate of 1/14 rather than 1/1. It affects 16% of
        # driver-seasons, always in the direction of flattering the newcomer.
        rounds_started = (
            pl.col("entered").cast(pl.Int32).shift(1).cum_sum().over(group).clip(lower_bound=1)
        )

        out = frame
        for source, target in _RATE_SOURCE_COLUMNS:
            out = out.with_columns(
                (pl.col(source).cast(pl.Int32).shift(1).cum_sum().over(group) / rounds_started)
                .fill_null(0.0)
                .alias(target)
            )

        return out
