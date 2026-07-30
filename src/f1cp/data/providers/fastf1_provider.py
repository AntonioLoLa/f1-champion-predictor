"""FastF1 provider: session timing data for 2018 onwards.

FastF1 exposes the official live-timing feed, which carries information no results table
contains: how fast a car actually was, independently of where it finished. A driver who
qualifies on pole and retires on lap 1 scores nothing, but their pace still says something
about their title chances.

Coverage starts in 2018 and each session is a multi-megabyte download, so this provider is
enrichment only and is disabled by default (``features.blocks.modern.enabled``). It cannot
reconstruct a championship on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from f1cp.data.identity import normalise_name
from f1cp.data.providers.base import Capability, EnrichmentProvider
from f1cp.domain.enums import ProviderName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from f1cp.config.settings import FastF1ProviderConfig

__all__ = ["FastF1Provider"]

logger = get_logger(__name__)

ENRICHMENT_COLUMNS = (
    "season",
    "round",
    "driver_id",
    "race_pace_median_ms",
    "race_pace_std_ms",
    "race_pace_delta_ms",
    "quali_best_ms",
    "stint_count",
)


class FastF1Provider(EnrichmentProvider):
    """Derives per-driver pace metrics from FastF1 session timing."""

    name: ClassVar[ProviderName] = ProviderName.FASTF1
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.ENRICHMENT})

    def __init__(
        self,
        config: FastF1ProviderConfig,
        *,
        driver_lookup: Mapping[str, str] | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            config: FastF1 settings.
            driver_lookup: Maps a normalised driver full name to a canonical driver id.
                FastF1 identifies drivers by three-letter abbreviation, which is reused
                across eras, so names are matched instead.
        """
        self._config = config
        self._driver_lookup = dict(driver_lookup or {})
        self._configured = False

    @property
    def min_season(self) -> int:
        """Earliest season with timing data (2018)."""
        return self._config.min_season

    def _ensure_configured(self) -> None:
        """Point FastF1 at the project cache directory, once per process."""
        if self._configured:
            return
        import fastf1

        self._config.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self._config.cache_dir))
        self._configured = True

    def fetch_enrichment(self, season: int, rounds: Sequence[int]) -> pl.DataFrame:
        """Compute pace metrics for the requested rounds.

        Args:
            season: Championship year.
            rounds: Rounds to load.

        Returns:
            One row per driver per round, with the columns in :data:`ENRICHMENT_COLUMNS`.
            Rounds that fail to load are skipped rather than aborting the whole run, since
            timing data for a recent session may simply not be published yet.
        """
        if season < self.min_season:
            logger.info(
                "FastF1 has no timing data for this season", extra={"season": season}
            )
            return pl.DataFrame()

        self._ensure_configured()

        frames = [
            frame
            for round_number in rounds
            if (frame := self._fetch_round(season, round_number)) is not None
        ]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _fetch_round(self, season: int, round_number: int) -> pl.DataFrame | None:
        """Load one race weekend, returning ``None`` if it is unavailable."""
        import fastf1

        try:
            session = fastf1.get_session(season, round_number, "R")
            session.load(telemetry=False, weather=False, messages=False)
        except Exception as exc:  # noqa: BLE001 - FastF1 raises a wide variety of errors
            logger.warning(
                "Skipping FastF1 session",
                extra={"season": season, "round": round_number, "error": str(exc)},
            )
            return None

        laps = session.laps
        if laps is None or laps.empty:
            return None

        # Exclude in-laps, out-laps and safety-car laps: they say nothing about pace.
        clean = laps.pick_wo_box() if hasattr(laps, "pick_wo_box") else laps
        clean = clean[clean["LapTime"].notna()]
        if clean.empty:
            return None

        milliseconds = clean["LapTime"].dt.total_seconds() * 1000.0
        working = pl.DataFrame(
            {
                "abbreviation": clean["Driver"].astype(str).tolist(),
                "lap_ms": milliseconds.tolist(),
                "stint": clean["Stint"].fillna(0).astype(int).tolist()
                if "Stint" in clean
                else [0] * len(clean),
            }
        )

        aggregated = working.group_by("abbreviation").agg(
            pl.col("lap_ms").median().alias("race_pace_median_ms"),
            pl.col("lap_ms").std().alias("race_pace_std_ms"),
            pl.col("stint").n_unique().alias("stint_count"),
        )

        # Absolute lap times are meaningless across circuits; the gap to the session's
        # median driver is what compares.
        session_median = aggregated["race_pace_median_ms"].median()
        aggregated = aggregated.with_columns(
            (pl.col("race_pace_median_ms") - session_median).alias("race_pace_delta_ms")
        )

        aggregated = aggregated.with_columns(
            pl.lit(season, dtype=pl.Int64).alias("season"),
            pl.lit(round_number, dtype=pl.Int64).alias("round"),
            pl.lit(None, dtype=pl.Float64).alias("quali_best_ms"),
            pl.col("abbreviation")
            .replace_strict(
                self._abbreviation_map(session), default=None, return_dtype=pl.String
            )
            .alias("driver_id"),
        )

        resolved = aggregated.filter(pl.col("driver_id").is_not_null())
        if resolved.height < aggregated.height:
            logger.warning(
                "Some FastF1 drivers could not be resolved to canonical ids",
                extra={
                    "season": season,
                    "round": round_number,
                    "unresolved": aggregated.height - resolved.height,
                },
            )
        return resolved.select(ENRICHMENT_COLUMNS)

    def _abbreviation_map(self, session: object) -> dict[str, str]:
        """Map this session's driver abbreviations onto canonical driver ids.

        Args:
            session: A loaded FastF1 session.

        Returns:
            Mapping of abbreviation to canonical driver id, omitting unresolved drivers.
        """
        results = getattr(session, "results", None)
        if results is None or results.empty:
            return {}

        mapping: dict[str, str] = {}
        for _, row in results.iterrows():
            abbreviation = str(row.get("Abbreviation", "")).strip()
            if not abbreviation:
                continue
            key = normalise_name(f"{row.get('FirstName', '')}{row.get('LastName', '')}")
            if canonical := self._driver_lookup.get(key):
                mapping[abbreviation] = canonical
        return mapping
