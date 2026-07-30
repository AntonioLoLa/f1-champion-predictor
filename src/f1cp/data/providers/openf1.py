"""OpenF1 provider: session data for 2023 onwards.

OpenF1 is a clean, key-free REST API over the live timing feed. Its coverage begins in
2023, which is eight seasons of data and three championships -- far too few positive labels
to train on, so it can only ever be an enrichment source. It is included because it is a
genuinely better interface than FastF1 for recent seasons: no multi-megabyte session
downloads, just JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from f1cp.data.http import HttpError, JsonHttpClient, RateLimit
from f1cp.data.providers.base import Capability, EnrichmentProvider
from f1cp.domain.enums import ProviderName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from f1cp.config.settings import OpenF1ProviderConfig
    from f1cp.data.http import ResponseCache

__all__ = ["OpenF1Provider"]

logger = get_logger(__name__)

ENRICHMENT_COLUMNS = (
    "season",
    "round",
    "driver_id",
    "openf1_lap_median_ms",
    "openf1_lap_best_ms",
    "openf1_pit_count",
)


class OpenF1Provider(EnrichmentProvider):
    """Derives per-driver lap metrics from the OpenF1 API."""

    name: ClassVar[ProviderName] = ProviderName.OPENF1
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.ENRICHMENT})

    def __init__(
        self,
        config: OpenF1ProviderConfig,
        *,
        cache: ResponseCache | None = None,
        driver_lookup: Mapping[str, str] | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            config: OpenF1 settings.
            cache: Optional shared response cache.
            driver_lookup: Maps a driver's normalised full name to a canonical driver id.
        """
        self._config = config
        self._driver_lookup = dict(driver_lookup or {})
        self._client = JsonHttpClient(
            config.base_url,
            rate_limit=RateLimit(config.requests_per_second, config.requests_per_hour),
            cache=cache,
            timeout=config.timeout_seconds,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()

    @property
    def min_season(self) -> int:
        """Earliest season with OpenF1 coverage (2023)."""
        return self._config.min_season

    def fetch_enrichment(self, season: int, rounds: Sequence[int]) -> pl.DataFrame:
        """Compute lap metrics for the requested rounds.

        Args:
            season: Championship year.
            rounds: Rounds to fetch. OpenF1 addresses weekends by meeting rather than by
                round, so the season's meetings are listed once and matched by order.

        Returns:
            One row per driver per round, with the columns in :data:`ENRICHMENT_COLUMNS`.
        """
        if season < self.min_season:
            logger.info("OpenF1 has no data for this season", extra={"season": season})
            return pl.DataFrame()

        try:
            meetings = self._client.get_json("meetings", params={"year": season})
        except HttpError as exc:
            logger.warning("Could not list OpenF1 meetings", extra={"error": str(exc)})
            return pl.DataFrame()

        if not isinstance(meetings, list) or not meetings:
            return pl.DataFrame()

        # Meetings come back in calendar order, which is round order.
        by_round = {index: meeting for index, meeting in enumerate(meetings, start=1)}

        frames = [
            frame
            for round_number in rounds
            if (meeting := by_round.get(round_number)) is not None
            and (frame := self._fetch_meeting(season, round_number, meeting)) is not None
        ]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _fetch_meeting(
        self, season: int, round_number: int, meeting: dict[str, Any]
    ) -> pl.DataFrame | None:
        """Fetch and aggregate the race session of one meeting."""
        meeting_key = meeting.get("meeting_key")
        if meeting_key is None:
            return None

        try:
            sessions = self._client.get_json(
                "sessions", params={"meeting_key": meeting_key, "session_name": "Race"}
            )
            if not isinstance(sessions, list) or not sessions:
                return None
            session_key = sessions[0].get("session_key")

            laps = self._client.get_json("laps", params={"session_key": session_key})
            drivers = self._client.get_json("drivers", params={"session_key": session_key})
            pits = self._client.get_json("pit", params={"session_key": session_key})
        except HttpError as exc:
            logger.warning(
                "Skipping OpenF1 meeting",
                extra={"season": season, "round": round_number, "error": str(exc)},
            )
            return None

        if not isinstance(laps, list) or not laps:
            return None

        lap_rows = [
            {
                "driver_number": lap.get("driver_number"),
                "lap_ms": float(lap["lap_duration"]) * 1000.0,
            }
            for lap in laps
            if lap.get("lap_duration") is not None and lap.get("driver_number") is not None
        ]
        if not lap_rows:
            return None

        aggregated = (
            pl.DataFrame(lap_rows)
            .group_by("driver_number")
            .agg(
                pl.col("lap_ms").median().alias("openf1_lap_median_ms"),
                pl.col("lap_ms").min().alias("openf1_lap_best_ms"),
            )
        )

        pit_counts = _count_pits(pits)
        number_to_driver = self._driver_number_map(drivers)

        return (
            aggregated.with_columns(
                pl.lit(season, dtype=pl.Int64).alias("season"),
                pl.lit(round_number, dtype=pl.Int64).alias("round"),
                pl.col("driver_number")
                .replace_strict(pit_counts, default=0, return_dtype=pl.Int64)
                .alias("openf1_pit_count"),
                pl.col("driver_number")
                .replace_strict(number_to_driver, default=None, return_dtype=pl.String)
                .alias("driver_id"),
            )
            .filter(pl.col("driver_id").is_not_null())
            .select(ENRICHMENT_COLUMNS)
        )

    def _driver_number_map(self, drivers: Any) -> dict[int, str]:
        """Map car numbers in this session onto canonical driver ids."""
        from f1cp.data.identity import normalise_name

        if not isinstance(drivers, list):
            return {}

        mapping: dict[int, str] = {}
        for entry in drivers:
            number = entry.get("driver_number")
            full_name = entry.get("full_name") or entry.get("broadcast_name") or ""
            if number is None:
                continue
            if canonical := self._driver_lookup.get(normalise_name(full_name)):
                mapping[int(number)] = canonical
        return mapping


def _count_pits(pits: Any) -> dict[int, int]:
    """Count pit stops per car number."""
    if not isinstance(pits, list):
        return {}
    counts: dict[int, int] = {}
    for stop in pits:
        if (number := stop.get("driver_number")) is not None:
            counts[int(number)] = counts.get(int(number), 0) + 1
    return counts
