"""Feature block protocol.

Each block is a small, independently testable unit that adds columns to the panel. Blocks
declare the columns they read so :mod:`f1cp.features.registry` can order them and fail
loudly if a dependency is missing, instead of a block silently computing nulls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["FeatureBlock"]


class FeatureBlock(ABC):
    """Base class for one group of related engineered features.

    Subclasses implement :meth:`build`, taking the frame accumulated by every block that
    ran before them and returning it with new columns appended. A block must never remove
    or overwrite a column it does not own, and must never look past the current round: see
    the module docstring of :mod:`f1cp.features.panel` for the ordering guarantee this
    relies on.
    """

    name: str
    """Unique identifier, matching the block's key in ``conf/features.yaml``."""

    requires: tuple[str, ...] = ()
    """Names of columns this block reads that it did not itself produce."""

    @abstractmethod
    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Compute this block's features.

        Args:
            frame: The panel as built so far, including every earlier block's columns.
            context: Shared inputs -- the source frames and feature configuration.

        Returns:
            ``frame`` with this block's columns appended.
        """
        raise NotImplementedError

    def check_requirements(self, available_columns: set[str]) -> None:
        """Fail fast if a declared dependency is missing.

        Args:
            available_columns: Columns produced by blocks that ran earlier.

        Raises:
            ValueError: If a required column is absent, which means the block ordering in
                :mod:`f1cp.features.registry` is wrong.
        """
        missing = [column for column in self.requires if column not in available_columns]
        if missing:
            msg = f"Feature block '{self.name}' is missing required columns: {missing}"
            raise ValueError(msg)
