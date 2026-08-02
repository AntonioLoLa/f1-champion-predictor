"""Kaggle provider: a static mirror of the retired Ergast database.

Included for completeness and as an offline fallback. It is **not** on the default
ingestion path, for two reasons measured during source research:

* the dataset stops at 2024, so it cannot serve the current championship;
* it requires Kaggle credentials, which makes it unusable in unauthenticated CI.

Its identifiers follow the Ergast vocabulary, so ingested data is passed through
:class:`~f1cp.data.identity.IdentityResolver` exactly like Jolpica's.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from f1cp.data.providers.base import BulkHistoryProvider, Capability, DataBundle, ProviderError
from f1cp.domain.enums import ProviderName, TableName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.config.settings import KaggleProviderConfig

__all__ = ["KaggleProvider"]

logger = get_logger(__name__)

_NULL_VALUES = ["\\N", ""]


class KaggleProvider(BulkHistoryProvider):
    """Loads the Ergast-derived Kaggle dataset."""

    name: ClassVar[ProviderName] = ProviderName.KAGGLE
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.BULK_HISTORY})

    def __init__(self, config: KaggleProviderConfig, *, download_dir: Path) -> None:
        self._config = config
        self._download_dir = download_dir

    def _download(self) -> Path:
        """Download the dataset via kagglehub.

        Returns:
            Path to the extracted dataset directory.

        Raises:
            ProviderError: If kagglehub is unavailable or credentials are missing.
        """
        try:
            import kagglehub
        except ImportError as exc:  # pragma: no cover - optional extra
            msg = "The Kaggle provider requires the 'kaggle' extra: pip install '.[kaggle]'"
            raise ProviderError(msg) from exc

        try:
            path = kagglehub.dataset_download(self._config.dataset)
        except Exception as exc:  # noqa: BLE001 - kagglehub raises bare exceptions
            msg = (
                f"Could not download {self._config.dataset!r}. Kaggle credentials "
                f"(KAGGLE_USERNAME / KAGGLE_KEY) are required. Original error: {exc}"
            )
            raise ProviderError(msg) from exc
        return Path(path)

    def _read(self, directory: Path, filename: str) -> pl.DataFrame:
        r"""Read one CSV of the dataset, tolerating its ``\N`` null marker."""
        path = directory / filename
        if not path.exists():
            logger.warning("Kaggle dataset is missing %s", filename)
            return pl.DataFrame()
        return pl.read_csv(path, infer_schema_length=0, null_values=_NULL_VALUES)

    def fetch_history(
        self,
        *,
        min_season: int | None = None,
        max_season: int | None = None,
    ) -> DataBundle:
        """Download and parse the dataset.

        Args:
            min_season: Earliest season to keep, or ``None``.
            max_season: Latest season to keep, or ``None``.

        Returns:
            Canonical tables in the Ergast identifier vocabulary.
        """
        directory = self._download()

        races = self._read(directory, "races.csv")
        circuits = self._read(directory, "circuits.csv")
        drivers = self._read(directory, "drivers.csv")
        constructors = self._read(directory, "constructors.csv")
        results = self._read(directory, "results.csv")
        qualifying = self._read(directory, "qualifying.csv")
        sprint_results = self._read(directory, "sprint_results.csv")
        driver_standings = self._read(directory, "driver_standings.csv")
        constructor_standings = self._read(directory, "constructor_standings.csv")
        status = self._read(directory, "status.csv")

        if not races.height:
            msg = "The Kaggle dataset did not contain races.csv"
            raise ProviderError(msg)

        # The Kaggle schema keys everything on surrogate integers; these lookups translate
        # them back into the (season, round) and slug identifiers used everywhere else.
        race_key = races.select(
            pl.col("raceId"),
            pl.col("year").cast(pl.Int64, strict=False).alias("season"),
            pl.col("round").cast(pl.Int64, strict=False).alias("round"),
        )
        driver_key = drivers.select(pl.col("driverId"), pl.col("driverRef").alias("driver_id"))
        constructor_key = constructors.select(
            pl.col("constructorId"), pl.col("constructorRef").alias("constructor_id")
        )
        circuit_key = circuits.select(pl.col("circuitId"), pl.col("circuitRef").alias("circuit_id"))
        status_key = (
            status.select(pl.col("statusId"), pl.col("status"))
            if status.height
            else pl.DataFrame(schema={"statusId": pl.String, "status": pl.String})
        )

        bundle = DataBundle.from_mapping(
            {
                TableName.CIRCUITS: circuits.select(
                    pl.col("circuitRef").alias("circuit_id"),
                    pl.col("name"),
                    pl.col("location").alias("locality"),
                    pl.col("country"),
                    pl.col("lat").cast(pl.Float64, strict=False).alias("latitude"),
                    pl.col("lng").cast(pl.Float64, strict=False).alias("longitude"),
                ),
                TableName.DRIVERS: drivers.select(
                    pl.col("driverRef").alias("driver_id"),
                    pl.col("code"),
                    pl.col("number").cast(pl.Int64, strict=False).alias("permanent_number"),
                    pl.col("forename").alias("given_name"),
                    pl.col("surname").alias("family_name"),
                    pl.col("dob").str.to_date(strict=False).alias("date_of_birth"),
                    pl.col("nationality"),
                ),
                TableName.CONSTRUCTORS: constructors.select(
                    pl.col("constructorRef").alias("constructor_id"),
                    pl.col("name"),
                    pl.col("nationality"),
                ),
                TableName.RACES: races.join(circuit_key, on="circuitId", how="left").select(
                    pl.col("year").cast(pl.Int64, strict=False).alias("season"),
                    pl.col("round").cast(pl.Int64, strict=False).alias("round"),
                    pl.col("name"),
                    pl.col("date").str.to_date(strict=False).alias("date"),
                    pl.col("circuit_id"),
                    pl.lit(value=False).alias("has_sprint"),
                ),
                TableName.RACE_RESULTS: _results_frame(
                    results, race_key, driver_key, constructor_key, status_key, sprint=False
                ),
                TableName.SPRINT_RESULTS: _results_frame(
                    sprint_results, race_key, driver_key, constructor_key, status_key, sprint=True
                ),
                TableName.QUALIFYING_RESULTS: _qualifying_frame(
                    qualifying, race_key, driver_key, constructor_key
                ),
                TableName.DRIVER_STANDINGS: _standings_frame(
                    driver_standings, race_key, driver_key, entity="driver_id"
                ),
                TableName.CONSTRUCTOR_STANDINGS: _standings_frame(
                    constructor_standings, race_key, constructor_key, entity="constructor_id"
                ),
            }
        )

        _mark_sprint_weekends(bundle)
        _restrict(bundle, min_season, max_season)
        logger.info("Parsed Kaggle dataset", extra={"rows": bundle.row_counts()})
        return bundle


def _results_frame(
    frame: pl.DataFrame,
    race_key: pl.DataFrame,
    driver_key: pl.DataFrame,
    constructor_key: pl.DataFrame,
    status_key: pl.DataFrame,
    *,
    sprint: bool,
) -> pl.DataFrame:
    """Translate a Kaggle results table into the canonical schema."""
    if not frame.height:
        return pl.DataFrame()

    joined = (
        frame.join(race_key, on="raceId", how="inner")
        .join(driver_key, on="driverId", how="left")
        .join(constructor_key, on="constructorId", how="left")
    )
    if status_key.height and "statusId" in joined.columns:
        joined = joined.join(status_key, on="statusId", how="left")

    columns = [
        pl.col("season"),
        pl.col("round"),
        pl.col("driver_id"),
        pl.col("constructor_id"),
        pl.col("grid").cast(pl.Int64, strict=False).alias("grid"),
        pl.col("position").cast(pl.Int64, strict=False).alias("position"),
        pl.col("positionOrder").cast(pl.Int64, strict=False).alias("position_order"),
        pl.col("points").cast(pl.Float64, strict=False).fill_null(0.0).alias("points"),
        pl.col("laps").cast(pl.Int64, strict=False).alias("laps"),
        (pl.col("status") if "status" in joined.columns else pl.lit(None)).alias("status"),
        pl.col("position").cast(pl.Int64, strict=False).is_not_null().alias("finished"),
    ]
    if not sprint:
        columns.append(pl.col("rank").cast(pl.Int64, strict=False).alias("fastest_lap_rank"))
    return joined.select(columns)


def _qualifying_frame(
    frame: pl.DataFrame,
    race_key: pl.DataFrame,
    driver_key: pl.DataFrame,
    constructor_key: pl.DataFrame,
) -> pl.DataFrame:
    """Translate the Kaggle qualifying table into the canonical schema."""
    if not frame.height:
        return pl.DataFrame()

    def _time_to_ms(column: str) -> pl.Expr:
        # Times arrive as "1:23.456"; split rather than regex-parse for clarity.
        minutes = pl.col(column).str.extract(r"^(\d+):", 1).cast(pl.Float64, strict=False)
        seconds = pl.col(column).str.extract(r":(\d+\.\d+)$", 1).cast(pl.Float64, strict=False)
        return ((minutes * 60 + seconds) * 1000).round().cast(pl.Int64, strict=False)

    return (
        frame.join(race_key, on="raceId", how="inner")
        .join(driver_key, on="driverId", how="left")
        .join(constructor_key, on="constructorId", how="left")
        .select(
            pl.col("season"),
            pl.col("round"),
            pl.col("driver_id"),
            pl.col("constructor_id"),
            pl.col("position").cast(pl.Int64, strict=False).alias("position"),
            _time_to_ms("q1").alias("q1_ms"),
            _time_to_ms("q2").alias("q2_ms"),
            _time_to_ms("q3").alias("q3_ms"),
        )
    )


def _standings_frame(
    frame: pl.DataFrame,
    race_key: pl.DataFrame,
    entity_key: pl.DataFrame,
    *,
    entity: str,
) -> pl.DataFrame:
    """Translate a Kaggle standings table into the canonical schema."""
    if not frame.height:
        return pl.DataFrame()

    join_column = "driverId" if entity == "driver_id" else "constructorId"
    return (
        frame.join(race_key, on="raceId", how="inner")
        .join(entity_key, on=join_column, how="left")
        .select(
            pl.col("season"),
            pl.col("round"),
            pl.col(entity),
            pl.col("points").cast(pl.Float64, strict=False).fill_null(0.0).alias("points"),
            pl.col("position").cast(pl.Int64, strict=False).alias("position"),
            pl.col("wins").cast(pl.Int64, strict=False).fill_null(0).alias("wins"),
        )
    )


def _mark_sprint_weekends(bundle: DataBundle) -> None:
    """Set ``has_sprint`` on the calendar from the presence of sprint results."""
    races = bundle.get(TableName.RACES)
    sprints = bundle.get(TableName.SPRINT_RESULTS)
    if not races.height or not sprints.height:
        return
    sprint_rounds = (
        sprints.select(["season", "round"])
        .unique()
        .with_columns(pl.lit(value=True).alias("is_sprint"))
    )
    bundle.tables[TableName.RACES] = (
        races.drop("has_sprint")
        .join(sprint_rounds, on=["season", "round"], how="left")
        .with_columns(pl.col("is_sprint").fill_null(value=False).alias("has_sprint"))
        .drop("is_sprint")
    )


def _restrict(bundle: DataBundle, min_season: int | None, max_season: int | None) -> None:
    """Filter every season-scoped table to the requested range."""
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
