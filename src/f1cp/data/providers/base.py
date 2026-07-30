"""Data provider abstraction.

The available sources differ in kind, not just in endpoint:

* **f1db** and **Kaggle** publish the whole history as one downloadable archive.
* **Jolpica** serves one season (or round) at a time over a rate-limited REST API.
* **FastF1** and **OpenF1** expose session telemetry for recent seasons only and cannot
  reconstruct a championship at all.

Forcing all of them behind one ``fetch()`` method would make the fast bulk path look like
the slow paginated one and would leave the enrichment providers implementing methods they
cannot honour. Instead there are three narrow roles, and a provider implements whichever
ones it can actually satisfy. All of them emit the same :class:`DataBundle`, which is the
real contract that makes the sources interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Self

import polars as pl

from f1cp.domain.enums import ProviderName, TableName

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "BulkHistoryProvider",
    "Capability",
    "DataBundle",
    "DataProvider",
    "EnrichmentProvider",
    "ProviderError",
    "SeasonProvider",
]


class ProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a request."""


class Capability(StrEnum):
    """What a provider is able to do."""

    BULK_HISTORY = "bulk_history"
    """Can download the full championship history in one shot."""

    SEASON_FETCH = "season_fetch"
    """Can fetch an individual season, suitable for incremental updates."""

    ENRICHMENT = "enrichment"
    """Can supply session-level detail (pace, telemetry) for recent seasons only."""


@dataclass(slots=True)
class DataBundle:
    """A set of canonical tables produced by a provider.

    Column names match the ORM models, so ingestion is a direct mapping rather than a
    per-provider translation step.
    """

    tables: dict[TableName, pl.DataFrame] = field(default_factory=dict)

    def get(self, table: TableName) -> pl.DataFrame:
        """Return a table, or an empty frame if the provider did not supply it.

        Args:
            table: Canonical table name.

        Returns:
            The table's data.
        """
        return self.tables.get(table, pl.DataFrame())

    def set(self, table: TableName, frame: pl.DataFrame) -> None:
        """Attach a table to the bundle.

        Args:
            table: Canonical table name.
            frame: The table's data. Empty frames are ignored.
        """
        if frame.height:
            self.tables[table] = frame

    def extend(self, other: DataBundle) -> None:
        """Vertically concatenate another bundle into this one.

        Used to accumulate per-season fetches into a single write.

        Args:
            other: Bundle to merge in.
        """
        for table, frame in other.tables.items():
            if not frame.height:
                continue
            existing = self.tables.get(table)
            if existing is None or not existing.height:
                self.tables[table] = frame
            else:
                self.tables[table] = pl.concat([existing, frame], how="diagonal_relaxed")

    def row_counts(self) -> dict[str, int]:
        """Row count per table, for logging and the ingestion audit record."""
        return {str(table): frame.height for table, frame in self.tables.items()}

    @property
    def total_rows(self) -> int:
        """Total number of rows across every table."""
        return sum(frame.height for frame in self.tables.values())

    @property
    def is_empty(self) -> bool:
        """Whether the bundle carries no rows at all."""
        return self.total_rows == 0

    @classmethod
    def from_mapping(cls, tables: Mapping[TableName, pl.DataFrame]) -> DataBundle:
        """Build a bundle from a mapping, skipping empty frames.

        Args:
            tables: Mapping of table name to data.

        Returns:
            A new bundle.
        """
        bundle = cls()
        for table, frame in tables.items():
            bundle.set(table, frame)
        return bundle


class DataProvider(ABC):
    """Base class for every data source."""

    name: ClassVar[ProviderName]
    capabilities: ClassVar[frozenset[Capability]]

    def supports(self, capability: Capability) -> bool:
        """Whether this provider offers a capability.

        Args:
            capability: The capability to check.

        Returns:
            True if supported.
        """
        return capability in self.capabilities

    def close(self) -> None:
        """Release any held resources. Overridden by providers holding connections."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class BulkHistoryProvider(DataProvider):
    """A provider that can download the complete history in one operation."""

    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.BULK_HISTORY})

    @abstractmethod
    def fetch_history(
        self,
        *,
        min_season: int | None = None,
        max_season: int | None = None,
    ) -> DataBundle:
        """Download the full history, optionally restricted to a season range.

        Args:
            min_season: Earliest season to include, or ``None`` for no lower bound.
            max_season: Latest season to include, or ``None`` for no upper bound.

        Returns:
            Every canonical table the source provides.
        """


class SeasonProvider(DataProvider):
    """A provider that serves one season at a time."""

    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.SEASON_FETCH})

    @abstractmethod
    def available_seasons(self) -> Sequence[int]:
        """Seasons this provider can serve, ascending."""

    @abstractmethod
    def fetch_season(self, season: int, *, rounds: Iterable[int] | None = None) -> DataBundle:
        """Fetch one season.

        Args:
            season: Championship year.
            rounds: Restrict to these rounds, or ``None`` for the whole season. Fetching
                only the missing rounds is what keeps incremental updates within the
                provider's rate limit.

        Returns:
            The season's canonical tables.
        """


class EnrichmentProvider(DataProvider):
    """A provider supplying session-level detail for recent seasons."""

    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.ENRICHMENT})

    @property
    @abstractmethod
    def min_season(self) -> int:
        """Earliest season for which this provider has data."""

    @abstractmethod
    def fetch_enrichment(self, season: int, rounds: Sequence[int]) -> pl.DataFrame:
        """Fetch per-driver, per-round enrichment metrics.

        Args:
            season: Championship year.
            rounds: Rounds to fetch.

        Returns:
            A frame keyed by ``(season, round, driver_id)`` with additional metrics.
        """
