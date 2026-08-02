"""f1db provider: the complete championship history as a single CSV release.

f1db publishes the whole of Formula 1 from 1950 onwards as a 4.5 MB archive, refreshed
within a day of each Grand Prix. Seeding from it turns a five-hour paginated backfill
against a rate-limited API into a single download, and its full-name slug identifiers are
the canonical vocabulary for the rest of the system (see :mod:`f1cp.data.identity`).

Two fields the canonical schema needs are not published directly and are derived here:

* ``wins`` in the standings tables -- f1db records points and position but not a win count,
  so it is accumulated from race results up to each round.
* ``constructor_id`` in the driver standings -- taken from the team the driver raced for
  that weekend.

Both are strictly backward-looking aggregations, so neither introduces future information.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import httpx
import polars as pl

from f1cp.data.providers.base import BulkHistoryProvider, Capability, DataBundle, ProviderError
from f1cp.domain.enums import ProviderName, TableName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.config.settings import F1dbProviderConfig

__all__ = ["F1dbProvider"]

logger = get_logger(__name__)

_RELEASE_DOWNLOAD_TEMPLATE = "https://github.com/f1db/f1db/releases/download/{tag}/{asset}"


def _read_csv(archive: zipfile.ZipFile, name: str) -> pl.DataFrame:
    """Read one CSV from the archive with every column as text.

    Columns such as ``gridPositionText`` mix integers and codes like ``"PL"``, so type
    inference is disabled and casts are applied explicitly downstream.

    Args:
        archive: The open release archive.
        name: File name inside the archive.

    Returns:
        The parsed frame, or an empty frame if the file is absent.
    """
    try:
        raw = archive.read(name)
    except KeyError:
        logger.warning("f1db release does not contain %s", name)
        return pl.DataFrame()
    return pl.read_csv(io.BytesIO(raw), infer_schema_length=0)


def _int(column: str) -> pl.Expr:
    """Cast a text column to a nullable integer."""
    return pl.col(column).cast(pl.Int64, strict=False)


def _float(column: str) -> pl.Expr:
    """Cast a text column to a float, treating blanks as zero."""
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)


class F1dbProvider(BulkHistoryProvider):
    """Loads the canonical championship history from an f1db CSV release."""

    name: ClassVar[ProviderName] = ProviderName.F1DB
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.BULK_HISTORY})

    def __init__(self, config: F1dbProviderConfig, *, download_dir: Path) -> None:
        self._config = config
        self._download_dir = download_dir
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_tag: str | None = None

    @property
    def release_tag(self) -> str | None:
        """The release tag actually used, once :meth:`fetch_history` has run."""
        return self._resolved_tag

    def _resolve_tag(self) -> str:
        """Determine which release to download.

        Returns:
            The pinned tag if configured, otherwise the latest published tag.

        Raises:
            ProviderError: If the latest release cannot be determined.
        """
        if self._config.pinned_tag:
            return self._config.pinned_tag

        try:
            response = httpx.get(
                self._config.latest_release_url,
                timeout=self._config.timeout_seconds,
                follow_redirects=True,
                headers={"Accept": "application/vnd.github+json"},
            )
            response.raise_for_status()
            tag = response.json().get("tag_name")
        except (httpx.HTTPError, ValueError) as exc:
            msg = f"Could not resolve the latest f1db release: {exc}"
            raise ProviderError(msg) from exc

        if not tag:
            msg = "The f1db release metadata contained no tag name"
            raise ProviderError(msg)
        return str(tag)

    def _download(self, tag: str) -> Path:
        """Download the release archive, reusing a previous download when present.

        Args:
            tag: Release tag, e.g. ``"v2026.11.0"``.

        Returns:
            Path to the archive on disk.

        Raises:
            ProviderError: If the download fails.
        """
        destination = self._download_dir / f"f1db-{tag}-{self._config.asset_name}"
        if destination.exists() and destination.stat().st_size > 0:
            logger.info("Reusing cached f1db release", extra={"tag": tag, "path": str(destination)})
            return destination

        url = _RELEASE_DOWNLOAD_TEMPLATE.format(tag=tag, asset=self._config.asset_name)
        logger.info("Downloading f1db release", extra={"tag": tag, "url": url})
        try:
            with httpx.stream(
                "GET", url, timeout=self._config.timeout_seconds, follow_redirects=True
            ) as response:
                response.raise_for_status()
                # Stream to a temporary file so an interrupted download is never mistaken
                # for a complete one on the next run.
                tmp = destination.with_suffix(".part")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
                tmp.replace(destination)
        except httpx.HTTPError as exc:
            msg = f"Failed to download the f1db release from {url}: {exc}"
            raise ProviderError(msg) from exc

        return destination

    def fetch_history(
        self,
        *,
        min_season: int | None = None,
        max_season: int | None = None,
    ) -> DataBundle:
        """Download and parse the full history.

        Args:
            min_season: Earliest season to keep, or ``None``.
            max_season: Latest season to keep, or ``None``.

        Returns:
            Every canonical table, restricted to the requested season range.

        Raises:
            ProviderError: If the archive cannot be downloaded or read.
        """
        tag = self._resolve_tag()
        self._resolved_tag = tag
        archive_path = self._download(tag)

        try:
            archive = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile as exc:
            # A corrupt cached download would otherwise fail on every subsequent run.
            archive_path.unlink(missing_ok=True)
            msg = f"The f1db archive at {archive_path} is corrupt and has been removed; retry"
            raise ProviderError(msg) from exc

        with archive:
            races = self._parse_races(archive)
            race_results = self._parse_results(archive, "f1db-races-race-results.csv", sprint=False)
            sprint_results = self._parse_results(
                archive, "f1db-races-sprint-race-results.csv", sprint=True
            )
            bundle = DataBundle.from_mapping(
                {
                    TableName.CIRCUITS: self._parse_circuits(archive),
                    TableName.DRIVERS: self._parse_drivers(archive),
                    TableName.CONSTRUCTORS: self._parse_constructors(archive),
                    TableName.RACES: races,
                    TableName.RACE_RESULTS: race_results,
                    TableName.SPRINT_RESULTS: sprint_results,
                    TableName.QUALIFYING_RESULTS: self._parse_qualifying(archive),
                    TableName.DRIVER_STANDINGS: self._parse_driver_standings(archive, race_results),
                    TableName.CONSTRUCTOR_STANDINGS: self._parse_constructor_standings(
                        archive, race_results
                    ),
                    TableName.PIT_STOPS: self._parse_pit_stops(archive),
                }
            )

        _restrict_seasons(bundle, min_season=min_season, max_season=max_season)
        logger.info(
            "Parsed f1db release",
            extra={"tag": tag, "rows": bundle.row_counts()},
        )
        return bundle

    @staticmethod
    def _parse_circuits(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-circuits.csv")
        if not frame.height:
            return frame
        return frame.select(
            pl.col("id").alias("circuit_id"),
            pl.coalesce(pl.col("fullName"), pl.col("name")).alias("name"),
            pl.col("placeName").alias("locality"),
            pl.col("countryId").alias("country"),
            pl.col("latitude").cast(pl.Float64, strict=False).alias("latitude"),
            pl.col("longitude").cast(pl.Float64, strict=False).alias("longitude"),
        )

    @staticmethod
    def _parse_drivers(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-drivers.csv")
        if not frame.height:
            return frame
        return frame.select(
            pl.col("id").alias("driver_id"),
            pl.col("abbreviation").alias("code"),
            _int("permanentNumber").alias("permanent_number"),
            pl.col("firstName").alias("given_name"),
            pl.col("lastName").alias("family_name"),
            pl.col("dateOfBirth").str.to_date(strict=False).alias("date_of_birth"),
            pl.col("nationalityCountryId").alias("nationality"),
        )

    @staticmethod
    def _parse_constructors(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-constructors.csv")
        if not frame.height:
            return frame
        return frame.select(
            pl.col("id").alias("constructor_id"),
            pl.col("name").alias("name"),
            pl.col("countryId").alias("nationality"),
        )

    @staticmethod
    def _parse_races(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-races.csv")
        if not frame.height:
            return frame
        return frame.select(
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.coalesce(pl.col("officialName"), pl.col("grandPrixId")).alias("name"),
            pl.col("date").str.to_date(strict=False).alias("date"),
            pl.col("circuitId").alias("circuit_id"),
            # A weekend is a sprint weekend exactly when a sprint race was run. Keyed off
            # the lap count rather than `sprintRaceDate`, which f1db only populates from
            # 2024 onwards even though sprints have been held since 2021.
            pl.col("sprintRaceLaps").is_not_null().alias("has_sprint"),
        )

    @staticmethod
    def _parse_results(archive: zipfile.ZipFile, filename: str, *, sprint: bool) -> pl.DataFrame:
        frame = _read_csv(archive, filename)
        if not frame.height:
            return pl.DataFrame()

        columns = [
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.col("driverId").alias("driver_id"),
            pl.col("constructorId").alias("constructor_id"),
            _int("gridPositionNumber").alias("grid"),
            _int("positionNumber").alias("position"),
            _int("positionDisplayOrder").alias("position_order"),
            _float("points").alias("points"),
            _int("laps").alias("laps"),
            pl.col("reasonRetired").alias("status"),
            # A driver is classified exactly when a finishing position was awarded.
            pl.col("positionNumber").is_not_null().alias("finished"),
        ]
        if not sprint:
            # f1db records the fastest lap as a flag rather than a rank; rank 1 is the only
            # value that ever mattered, since only it scored the bonus point (2019-2024).
            columns.append(
                pl.when(pl.col("fastestLap") == "true")
                .then(pl.lit(1, dtype=pl.Int64))
                .otherwise(None)
                .alias("fastest_lap_rank")
            )
        return frame.select(columns)

    @staticmethod
    def _parse_qualifying(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-races-qualifying-results.csv")
        if not frame.height:
            return frame
        return frame.select(
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.col("driverId").alias("driver_id"),
            pl.col("constructorId").alias("constructor_id"),
            _int("positionNumber").alias("position"),
            _int("q1Millis").alias("q1_ms"),
            _int("q2Millis").alias("q2_ms"),
            _int("q3Millis").alias("q3_ms"),
        )

    @staticmethod
    def _parse_pit_stops(archive: zipfile.ZipFile) -> pl.DataFrame:
        frame = _read_csv(archive, "f1db-races-pit-stops.csv")
        if not frame.height:
            return frame
        return frame.select(
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.col("driverId").alias("driver_id"),
            _int("stop").alias("stop"),
            _int("lap").alias("lap"),
            _int("timeMillis").alias("duration_ms"),
        )

    @staticmethod
    def _parse_driver_standings(
        archive: zipfile.ZipFile, race_results: pl.DataFrame
    ) -> pl.DataFrame:
        """Parse driver standings, deriving the win count and the constructor."""
        frame = _read_csv(archive, "f1db-races-driver-standings.csv")
        if not frame.height:
            return frame

        standings = frame.select(
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.col("driverId").alias("driver_id"),
            _float("points").alias("points"),
            _int("positionNumber").alias("position"),
        )
        return (
            standings.join(
                _cumulative_wins(race_results, entity="driver_id"),
                on=["season", "round", "driver_id"],
                how="left",
            )
            .join(
                _constructor_by_round(race_results),
                on=["season", "round", "driver_id"],
                how="left",
            )
            .with_columns(pl.col("wins").fill_null(0))
        )

    @staticmethod
    def _parse_constructor_standings(
        archive: zipfile.ZipFile, race_results: pl.DataFrame
    ) -> pl.DataFrame:
        """Parse constructor standings, deriving the win count."""
        frame = _read_csv(archive, "f1db-races-constructor-standings.csv")
        if not frame.height:
            return frame

        standings = frame.select(
            _int("year").alias("season"),
            _int("round").alias("round"),
            pl.col("constructorId").alias("constructor_id"),
            _float("points").alias("points"),
            _int("positionNumber").alias("position"),
        )
        return standings.join(
            _cumulative_wins(race_results, entity="constructor_id"),
            on=["season", "round", "constructor_id"],
            how="left",
        ).with_columns(pl.col("wins").fill_null(0))


def _cumulative_wins(race_results: pl.DataFrame, *, entity: str) -> pl.DataFrame:
    """Count wins accumulated by each entity up to and including every round.

    Args:
        race_results: Canonical race results.
        entity: ``"driver_id"`` or ``"constructor_id"``.

    Returns:
        A frame of ``(season, round, entity, wins)``.
    """
    if not race_results.height:
        return pl.DataFrame(
            schema={"season": pl.Int64, "round": pl.Int64, entity: pl.String, "wins": pl.Int64}
        )

    # One team can win only once per race, so wins are counted per race first.
    per_round = (
        race_results.filter(pl.col("position") == 1)
        .select(["season", "round", entity])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("win"))
    )
    # Every (season, round, entity) combination needs a row, not just the winning ones,
    # so the cumulative count is defined for every standings row.
    scaffold = race_results.select(["season", "round", entity]).unique()
    return (
        scaffold.join(per_round, on=["season", "round", entity], how="left")
        .with_columns(pl.col("win").fill_null(0))
        .sort(["season", entity, "round"])
        .with_columns(pl.col("win").cum_sum().over(["season", entity]).alias("wins"))
        .select(["season", "round", entity, "wins"])
    )


def _constructor_by_round(race_results: pl.DataFrame) -> pl.DataFrame:
    """Map each driver to the constructor they raced for in a given round."""
    if not race_results.height:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "round": pl.Int64,
                "driver_id": pl.String,
                "constructor_id": pl.String,
            }
        )
    return race_results.select(["season", "round", "driver_id", "constructor_id"]).unique(
        subset=["season", "round", "driver_id"], keep="first"
    )


def _restrict_seasons(
    bundle: DataBundle, *, min_season: int | None, max_season: int | None
) -> None:
    """Filter every season-scoped table in place to the requested range."""
    if min_season is None and max_season is None:
        return
    for table, frame in list(bundle.tables.items()):
        if "season" not in frame.columns:
            continue
        filtered = frame
        if min_season is not None:
            filtered = filtered.filter(pl.col("season") >= min_season)
        if max_season is not None:
            filtered = filtered.filter(pl.col("season") <= max_season)
        bundle.tables[table] = filtered
