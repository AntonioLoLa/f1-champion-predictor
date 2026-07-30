"""Inputs shared by every feature block.

The feature layer is deliberately pure: it receives polars frames and returns polars
frames, and imports nothing from :mod:`f1cp.db`. Loading the frames out of the database is
:mod:`f1cp.dataset.loader`'s job. That separation is what lets the whole feature suite be
tested against hand-written frames, with no database and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from f1cp.config.settings import FeaturesConfig, PathsConfig

__all__ = ["KEY_COLUMNS", "FeatureContext", "SourceFrames"]

KEY_COLUMNS: tuple[str, ...] = ("season", "round", "driver_id")
"""Grain of the modelling dataset: one row per driver per completed round."""


@dataclass(frozen=True, slots=True)
class SourceFrames:
    """Curated Formula 1 data, one frame per table.

    Every frame is keyed by ``season`` and ``round`` rather than by the database's
    surrogate ``race_id``, so the feature layer never has to know how rows are stored.

    Attributes:
        races: Full season calendars, including rounds not yet run.
        race_results: Grand Prix results.
        sprint_results: Sprint race results (2021 onwards).
        qualifying: Qualifying results.
        driver_standings: Provider-supplied championship standings after each round.
        constructor_standings: Provider-supplied constructors' standings after each round.
        drivers: Driver reference data.
        constructors: Constructor reference data.
        circuits: Circuit reference data.
    """

    races: pl.DataFrame
    race_results: pl.DataFrame
    sprint_results: pl.DataFrame
    qualifying: pl.DataFrame
    driver_standings: pl.DataFrame
    constructor_standings: pl.DataFrame
    drivers: pl.DataFrame
    constructors: pl.DataFrame
    circuits: pl.DataFrame

    def as_of(self, season: int, round_number: int) -> SourceFrames:
        """Return a copy with every observation after ``(season, round_number)`` removed.

        This is the mechanism behind the leakage test suite: a feature computed for
        ``(season, R)`` from the truncated sources must equal the value computed from the
        full sources. The calendar is *not* truncated, because a season's remaining
        schedule is known in advance and is legitimate information at round ``R``.

        Args:
            season: Championship year to truncate at.
            round_number: Last round whose results remain visible.

        Returns:
            A new :class:`SourceFrames` holding only information available at that point.
        """

        def cut(frame: pl.DataFrame) -> pl.DataFrame:
            if not frame.height or "season" not in frame.columns:
                return frame
            return frame.filter(
                (pl.col("season") < season)
                | ((pl.col("season") == season) & (pl.col("round") <= round_number))
            )

        return SourceFrames(
            races=self.races,
            race_results=cut(self.race_results),
            sprint_results=cut(self.sprint_results),
            qualifying=cut(self.qualifying),
            driver_standings=cut(self.driver_standings),
            constructor_standings=cut(self.constructor_standings),
            drivers=self.drivers,
            constructors=self.constructors,
            circuits=self.circuits,
        )


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Everything a feature block needs to do its work.

    Attributes:
        panel: The dense ``(season, round, driver_id)`` panel produced by
            :func:`f1cp.features.panel.build_panel`. Sorted by driver then round, so
            rolling and cumulative window expressions are well defined.
        sources: The underlying curated frames, for blocks that need more than the panel.
        config: Feature engineering configuration.
        paths: Filesystem layout, for blocks that read side files such as the enrichment
            Parquet written by ``f1cp data enrich``.
    """

    panel: pl.DataFrame
    sources: SourceFrames
    config: FeaturesConfig
    paths: PathsConfig

    @property
    def keys(self) -> list[str]:
        """The key columns of the dataset grain."""
        return list(KEY_COLUMNS)

    def key_frame(self) -> pl.DataFrame:
        """The panel reduced to its key columns, preserving order."""
        return self.panel.select(self.keys)
