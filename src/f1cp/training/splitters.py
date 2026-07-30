"""Season-level, walk-forward cross-validation.

Never a random split, and never a split within a season: a season's rounds are strongly
autocorrelated (the same driver's championship state on round 10 and round 11 are nearly
identical), so anything but splitting on whole seasons would leak. Both splitters here
train on past seasons and test on the season(s) immediately following, which is the only
validation scheme that mirrors how the model is actually used: predicting a season nobody
has seen the outcome of yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import polars as pl

__all__ = [
    "Fold",
    "RollingWindowSplit",
    "SeasonForwardSplit",
    "Splitter",
    "completed_seasons",
]


def completed_seasons(frame: pl.DataFrame) -> list[int]:
    """Seasons in ``frame`` whose full calendar has been played out.

    A season still in progress has no reliable label (every row reads ``champion = 0``
    because nobody has won yet), so it must never be used as a training or test season --
    only as an inference target. A season counts as complete when its last stored round
    equals ``rounds_total``.

    Args:
        frame: The built dataset, carrying ``season``, ``round`` and ``rounds_total``.

    Returns:
        Completed seasons, ascending.
    """
    last_round = frame.group_by("season").agg(
        pl.col("round").max().alias("_last_round"),
        pl.col("rounds_total").max().alias("_rounds_total"),
    )
    return sorted(
        last_round.filter(pl.col("_last_round") == pl.col("_rounds_total"))["season"].to_list()
    )


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward split.

    Attributes:
        train_seasons: Seasons used to fit the model, ascending.
        test_seasons: Seasons used to evaluate it, ascending, all strictly after every
            training season.
    """

    train_seasons: list[int]
    test_seasons: list[int]

    def __post_init__(self) -> None:
        """Validate that train and test seasons don't overlap and are chronologically ordered."""
        overlap = set(self.train_seasons) & set(self.test_seasons)
        if overlap:
            msg = f"Fold has overlapping train/test seasons: {sorted(overlap)}"
            raise ValueError(msg)
        if (
            self.train_seasons
            and self.test_seasons
            and max(self.train_seasons) >= min(self.test_seasons)
        ):
            msg = "Fold's test seasons must all come after every training season"
            raise ValueError(msg)


class Splitter(ABC):
    """Produces walk-forward folds over a sequence of completed seasons."""

    @abstractmethod
    def split(self, seasons: list[int]) -> Iterator[Fold]:
        """Yield folds in chronological order.

        Args:
            seasons: Completed seasons available to split, any order.

        Yields:
            Folds whose test seasons strictly follow their training seasons.
        """
        raise NotImplementedError


class SeasonForwardSplit(Splitter):
    """Expanding-origin split: train on every season seen so far, test on the next.

    The training window grows by one step each fold, matching how the model is actually
    retrained in production -- on everything known up to today.
    """

    def __init__(self, min_train_seasons: int, test_horizon_seasons: int = 1) -> None:
        """Configure the split.

        Args:
            min_train_seasons: Fewest seasons required before the first fold is emitted.
            test_horizon_seasons: How many seasons each fold tests on.
        """
        self._min_train_seasons = min_train_seasons
        self._test_horizon_seasons = test_horizon_seasons

    def split(self, seasons: list[int]) -> Iterator[Fold]:
        """See :meth:`Splitter.split`."""
        ordered = sorted(seasons)
        cursor = self._min_train_seasons
        while cursor + self._test_horizon_seasons <= len(ordered):
            yield Fold(
                train_seasons=ordered[:cursor],
                test_seasons=ordered[cursor : cursor + self._test_horizon_seasons],
            )
            cursor += self._test_horizon_seasons


class RollingWindowSplit(Splitter):
    """Fixed-width trailing window: train on the last N seasons, test on the next.

    Unlike :class:`SeasonForwardSplit`, old seasons eventually fall out of the training
    window -- useful for measuring whether a model trained only on recent regulation eras
    generalises better than one trained on the full, era-spanning history.
    """

    def __init__(self, window_seasons: int, test_horizon_seasons: int = 1) -> None:
        """Configure the split.

        Args:
            window_seasons: Number of seasons in the trailing training window.
            test_horizon_seasons: How many seasons each fold tests on.
        """
        self._window_seasons = window_seasons
        self._test_horizon_seasons = test_horizon_seasons

    def split(self, seasons: list[int]) -> Iterator[Fold]:
        """See :meth:`Splitter.split`."""
        ordered = sorted(seasons)
        cursor = self._window_seasons
        while cursor + self._test_horizon_seasons <= len(ordered):
            yield Fold(
                train_seasons=ordered[cursor - self._window_seasons : cursor],
                test_seasons=ordered[cursor : cursor + self._test_horizon_seasons],
            )
            cursor += self._test_horizon_seasons
