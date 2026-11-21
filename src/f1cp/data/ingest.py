"""Ingestion orchestration.

Turns provider output into rows in the curated tables. Three modes:

``bootstrap``
    Seed the whole history from a bulk release. Seconds, one download.
``incremental``
    Fetch only rounds that have been run but are not yet stored. Roughly seven API
    requests per new round, which is what keeps the pipeline inside Jolpica's budget.
``refresh``
    Re-fetch a specific season, for when official results are amended after the fact.

Every write is an upsert, so any mode can be re-run safely.

Foreign-vocabulary providers (Jolpica, Kaggle) are passed through
:class:`~f1cp.data.identity.IdentityResolver` before writing, so a single canonical
identifier per driver, constructor and circuit reaches the database.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from sqlalchemy import inspect as sa_inspect

from f1cp.data.identity import IdentityResolver
from f1cp.data.providers.base import Capability, DataBundle, ProviderError
from f1cp.db.models import (
    Circuit,
    Constructor,
    ConstructorStanding,
    Driver,
    DriverStanding,
    PitStop,
    QualifyingResult,
    Race,
    RaceResult,
    SprintResult,
)
from f1cp.domain.enums import IngestionMode, ProviderName, RunStatus, TableName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from f1cp.config.settings import Settings
    from f1cp.data.providers.registry import ProviderRegistry
    from f1cp.db.session import Database
    from f1cp.db.uow import UnitOfWork

__all__ = ["DataIngestor", "IngestionReport"]

logger = get_logger(__name__)

# Providers whose identifiers must be translated into the canonical f1db vocabulary.
_FOREIGN_VOCABULARY_PROVIDERS = frozenset({ProviderName.JOLPICA, ProviderName.KAGGLE})

# Order matters: parents before children, so foreign keys always resolve.
_REFERENCE_TABLES: tuple[tuple[TableName, type[Any]], ...] = (
    (TableName.CIRCUITS, Circuit),
    (TableName.DRIVERS, Driver),
    (TableName.CONSTRUCTORS, Constructor),
)
_RACE_SCOPED_TABLES: tuple[tuple[TableName, type[Any]], ...] = (
    (TableName.RACE_RESULTS, RaceResult),
    (TableName.SPRINT_RESULTS, SprintResult),
    (TableName.QUALIFYING_RESULTS, QualifyingResult),
    (TableName.DRIVER_STANDINGS, DriverStanding),
    (TableName.CONSTRUCTOR_STANDINGS, ConstructorStanding),
    (TableName.PIT_STOPS, PitStop),
)


@dataclass(slots=True)
class IngestionReport:
    """Outcome of one ingestion run.

    Attributes:
        provider: Provider that supplied the data.
        mode: Which ingestion mode ran.
        seasons: Seasons touched.
        rows_written: Row counts per table.
        skipped_rows: Rows dropped because a required field was missing.
        status: Final status.
        error: Failure message, if any.
    """

    provider: str
    mode: IngestionMode
    seasons: list[int] = field(default_factory=list)
    rows_written: dict[str, int] = field(default_factory=dict)
    skipped_rows: dict[str, int] = field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    error: str | None = None

    @property
    def total_rows(self) -> int:
        """Total rows written across every table."""
        return sum(self.rows_written.values())

    def merge(self, other: IngestionReport) -> None:
        """Accumulate another report's counts into this one.

        Args:
            other: Report to fold in.
        """
        self.seasons = sorted(set(self.seasons) | set(other.seasons))
        for table, count in other.rows_written.items():
            self.rows_written[table] = self.rows_written.get(table, 0) + count
        for table, count in other.skipped_rows.items():
            self.skipped_rows[table] = self.skipped_rows.get(table, 0) + count


class DataIngestor:
    """Coordinates providers, identity reconciliation and database writes."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        registry: ProviderRegistry,
    ) -> None:
        self._settings = settings
        self._database = database
        self._registry = registry

    def bootstrap(
        self,
        *,
        min_season: int | None = None,
        max_season: int | None = None,
        provider_name: str | None = None,
    ) -> IngestionReport:
        """Seed the database from a bulk history release.

        Args:
            min_season: Earliest season to load. Defaults to loading everything, because
                career-level features look further back than the modelling window.
            max_season: Latest season to load.
            provider_name: Override the configured bulk provider.

        Returns:
            A populated :class:`IngestionReport`.
        """
        provider = self._registry.bulk(provider_name)
        report = IngestionReport(provider=str(provider.name), mode=IngestionMode.BOOTSTRAP)

        with self._audit(report):
            with provider:
                bundle = provider.fetch_history(min_season=min_season, max_season=max_season)
            if bundle.is_empty:
                logger.warning("Bulk provider returned no data", extra={"provider": provider.name})
                return report

            self._apply_identity_if_needed(provider.name, bundle)
            with self._database.session_scope() as uow:
                self._write(bundle, uow, report)
            report.seasons = _seasons_in(bundle)

        return report

    def incremental(
        self,
        *,
        seasons: Sequence[int] | None = None,
        provider_name: str | None = None,
    ) -> IngestionReport:
        """Fetch rounds that have been run but are not yet stored.

        Args:
            seasons: Seasons to check. Defaults to the current season only, which is the
                only one that can gain new rounds.
            provider_name: Override the configured incremental provider.

        Returns:
            A populated :class:`IngestionReport`.
        """
        provider = self._registry.season(provider_name)
        report = IngestionReport(provider=str(provider.name), mode=IngestionMode.INCREMENTAL)
        target_seasons = list(seasons) if seasons else [dt.date.today().year]

        with self._audit(report), provider:
            for season in target_seasons:
                missing = self._missing_rounds(season)
                if missing is None:
                    # A season the database has never seen -- every new year, once the
                    # bootstrap's last season is over. Only a whole-season fetch brings in
                    # its calendar; until then there is nothing to compare dates against.
                    logger.info("Fetching a season not yet stored", extra={"season": season})
                elif not missing:
                    logger.info("Season is already up to date", extra={"season": season})
                    continue
                else:
                    logger.info(
                        "Fetching missing rounds",
                        extra={"season": season, "rounds": missing},
                    )
                bundle = provider.fetch_season(season, rounds=missing)
                if bundle.is_empty:
                    continue

                self._apply_identity_if_needed(provider.name, bundle)
                with self._database.session_scope() as uow:
                    self._write(bundle, uow, report)
                report.seasons.append(season)

        return report

    def refresh_season(
        self,
        season: int,
        *,
        provider_name: str | None = None,
    ) -> IngestionReport:
        """Re-fetch an entire season, overwriting what is stored.

        Args:
            season: Championship year.
            provider_name: Override the configured incremental provider.

        Returns:
            A populated :class:`IngestionReport`.
        """
        provider = self._registry.season(provider_name)
        report = IngestionReport(
            provider=str(provider.name), mode=IngestionMode.REFRESH, seasons=[season]
        )

        with self._audit(report):
            with provider:
                bundle = provider.fetch_season(season)
            if bundle.is_empty:
                return report
            self._apply_identity_if_needed(provider.name, bundle)
            with self._database.session_scope() as uow:
                self._write(bundle, uow, report)

        return report

    def enrich(
        self,
        provider_name: str,
        *,
        seasons: Sequence[int] | None = None,
    ) -> pl.DataFrame:
        """Fetch enrichment metrics from a session-data provider.

        Enrichment is not written to the curated tables: it is optional, sparse and only
        covers recent seasons, so it is materialised alongside the feature matrix instead.

        Args:
            provider_name: Enrichment provider name.
            seasons: Seasons to enrich. Defaults to every season the provider covers.

        Returns:
            A frame keyed by ``(season, round, driver_id)``.

        Raises:
            ProviderError: If the provider offers no enrichment capability.
        """
        with self._database.session_scope() as uow:
            lookup = _driver_name_lookup(uow)
            stored_seasons = list(uow.races.seasons())

        provider = self._registry.enrichment(provider_name, driver_lookup=lookup)
        if not provider.supports(Capability.ENRICHMENT):  # pragma: no cover - defensive
            msg = f"{provider_name} offers no enrichment"
            raise ProviderError(msg)

        candidates = list(seasons) if seasons else stored_seasons
        targets = [season for season in candidates if season >= provider.min_season]

        frames: list[pl.DataFrame] = []
        with provider, self._database.session_scope() as uow:
            for season in targets:
                rounds = list(uow.races.rounds_with_results(season))
                if not rounds:
                    continue
                frame = provider.fetch_enrichment(season, rounds)
                if frame.height:
                    frames.append(frame)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _missing_rounds(self, season: int) -> list[int] | None:
        """Rounds of a season that have taken place but have no stored results.

        A round counts as run when its scheduled date is in the past. Comparing against
        the stored calendar avoids an API call just to discover there is nothing to do.

        A season with no stored calendar at all is a different case from one that is up to
        date, and must not be reported as an empty list: the scheduled data sync runs
        ``incremental()`` for the current year only, so treating an unknown season as
        "nothing to do" would leave every season after the bootstrap unfetched, silently.

        Args:
            season: Championship year.

        Returns:
            Ascending round numbers still to fetch, or ``None`` if the season is not
            stored and has to be fetched whole.
        """
        today = dt.date.today()
        with self._database.session_scope() as uow:
            stored = set(uow.races.rounds_with_results(season))
            calendar = {race.round: race.date for race in uow.races.list() if race.season == season}

        if not calendar:
            return None

        return sorted(
            round_number
            for round_number, race_date in calendar.items()
            if round_number not in stored and race_date is not None and race_date <= today
        )

    def _apply_identity_if_needed(self, provider_name: ProviderName, bundle: DataBundle) -> None:
        """Translate a foreign provider's identifiers into the canonical vocabulary."""
        if provider_name not in _FOREIGN_VOCABULARY_PROVIDERS:
            return

        seasons = _seasons_in(bundle) or _seasons_in_results(bundle)
        with self._database.session_scope() as uow:
            canonical_drivers = _frame_from(uow.drivers.list(), Driver)
            canonical_constructors = _frame_from(uow.constructors.list(), Constructor)
            canonical_circuits = _frame_from(uow.circuits.list(), Circuit)
            canonical_lineups = _stored_lineups(uow, seasons)

        resolver = IdentityResolver.build(
            canonical_drivers=canonical_drivers,
            canonical_constructors=canonical_constructors,
            canonical_circuits=canonical_circuits,
            foreign_drivers=bundle.get(TableName.DRIVERS),
            foreign_constructors=bundle.get(TableName.CONSTRUCTORS),
            foreign_circuits=bundle.get(TableName.CIRCUITS),
            canonical_lineups=canonical_lineups,
            foreign_lineups=_bundle_lineups(bundle),
        )
        resolver.apply(bundle)
        logger.info(
            "Reconciled provider identifiers",
            extra={"provider": str(provider_name), **resolver.coverage()},
        )

    def _write(self, bundle: DataBundle, uow: UnitOfWork, report: IngestionReport) -> None:
        """Write a bundle to the database, parents first."""
        for table, model in _REFERENCE_TABLES:
            rows, skipped = _align(bundle.get(table), model)
            if rows:
                written = _upsert_reference(uow, table, rows)
                report.rows_written[table] = report.rows_written.get(table, 0) + written
            if skipped:
                report.skipped_rows[table] = report.skipped_rows.get(table, 0) + skipped

        races = bundle.get(TableName.RACES)
        race_rows, skipped = _align(races, Race)
        if race_rows:
            written = uow.races.upsert_many(race_rows)
            report.rows_written[TableName.RACES] = (
                report.rows_written.get(TableName.RACES, 0) + written
            )
        if skipped:
            report.skipped_rows[TableName.RACES] = skipped

        # Races must be committed before their surrogate ids can be looked up.
        uow.commit()
        race_ids = uow.races.id_map()

        for table, model in _RACE_SCOPED_TABLES:
            frame = _attach_race_id(bundle.get(table), race_ids)
            rows, skipped = _align(frame, model)
            if rows:
                written = _upsert_race_scoped(uow, table, rows)
                report.rows_written[table] = report.rows_written.get(table, 0) + written
            if skipped:
                report.skipped_rows[table] = report.skipped_rows.get(table, 0) + skipped

        uow.commit()

    def _audit(self, report: IngestionReport) -> _AuditContext:
        """Context manager recording the run in ``data_ingestions``."""
        return _AuditContext(self._database, report)


class _AuditContext:
    """Writes an audit row for an ingestion run, whatever its outcome."""

    def __init__(self, database: Database, report: IngestionReport) -> None:
        self._database = database
        self._report = report
        self._started = dt.datetime.now(dt.UTC)

    def __enter__(self) -> IngestionReport:
        self._report.status = RunStatus.RUNNING
        return self._report

    def __exit__(
        self, exc_type: object, exc_value: BaseException | None, _tb: object
    ) -> Literal[False]:
        if exc_value is None:
            self._report.status = RunStatus.SUCCEEDED
        else:
            self._report.status = RunStatus.FAILED
            self._report.error = str(exc_value)

        from f1cp.db.models import DataIngestion

        with self._database.session_scope() as uow:
            uow.ingestions.add(
                DataIngestion(
                    provider=self._report.provider,
                    mode=str(self._report.mode),
                    status=str(self._report.status),
                    seasons_covered=_summarise_seasons(self._report.seasons),
                    rows_written=self._report.total_rows,
                    row_counts=dict(self._report.rows_written),
                    started_at=self._started,
                    finished_at=dt.datetime.now(dt.UTC),
                    error=self._report.error,
                )
            )
        # Never swallow the original exception.
        return False


def _upsert_reference(uow: UnitOfWork, table: TableName, rows: list[dict[str, Any]]) -> int:
    match table:
        case TableName.CIRCUITS:
            return uow.circuits.upsert_many(rows)
        case TableName.DRIVERS:
            return uow.drivers.upsert_many(rows)
        case TableName.CONSTRUCTORS:
            return uow.constructors.upsert_many(rows)
        case _:  # pragma: no cover - table list is closed
            return 0


def _upsert_race_scoped(uow: UnitOfWork, table: TableName, rows: list[dict[str, Any]]) -> int:
    match table:
        case TableName.RACE_RESULTS:
            return uow.results.upsert_race_results(rows)
        case TableName.SPRINT_RESULTS:
            return uow.results.upsert_sprint_results(rows)
        case TableName.QUALIFYING_RESULTS:
            return uow.results.upsert_qualifying(rows)
        case TableName.PIT_STOPS:
            return uow.results.upsert_pit_stops(rows)
        case TableName.DRIVER_STANDINGS:
            return uow.standings.upsert_driver_standings(rows)
        case TableName.CONSTRUCTOR_STANDINGS:
            return uow.standings.upsert_constructor_standings(rows)
        case _:  # pragma: no cover - table list is closed
            return 0


def _attach_race_id(frame: pl.DataFrame, race_ids: dict[tuple[int, int], int]) -> pl.DataFrame:
    """Replace ``(season, round)`` with the surrogate ``race_id``.

    Rows referencing a race that is not on the stored calendar are dropped: without a
    parent row they would violate the foreign key.

    Args:
        frame: A race-scoped table keyed by season and round.
        race_ids: Mapping produced by :meth:`RaceRepository.id_map`.

    Returns:
        The frame with a ``race_id`` column and no season/round columns.
    """
    if not frame.height or "season" not in frame.columns or "round" not in frame.columns:
        return frame

    keys = pl.DataFrame(
        {
            "season": [season for season, _ in race_ids],
            "round": [round_number for _, round_number in race_ids],
            "race_id": list(race_ids.values()),
        },
        schema={"season": pl.Int64, "round": pl.Int64, "race_id": pl.Int64},
    )
    return frame.join(keys, on=["season", "round"], how="inner").drop("season", "round")


def _align(frame: pl.DataFrame, model: type[Any]) -> tuple[list[dict[str, Any]], int]:
    """Project a frame onto a model's columns and drop rows missing required values.

    The ORM is the single source of truth for what a table looks like, so provider output
    is validated against it rather than against a duplicated schema definition.

    Args:
        frame: Provider output.
        model: Target ORM model.

    Returns:
        A tuple of writable row dictionaries and the number of rows dropped.
    """
    if not frame.height:
        return [], 0

    mapper = sa_inspect(model)
    columns = {column.key: column for column in mapper.columns}
    # Surrogate integer keys come from the database, never from a provider. The check is
    # `is True` rather than truthiness because SQLAlchemy defaults `autoincrement` to the
    # string "auto" on natural string primary keys such as `circuit_id`.
    writable = [
        name
        for name in frame.columns
        if name in columns
        and not (columns[name].primary_key and columns[name].autoincrement is True)
    ]
    if not writable:
        return [], frame.height

    required = [
        name for name in writable if not columns[name].nullable and columns[name].default is None
    ]

    projected = frame.select(writable)
    before = projected.height
    if required:
        projected = projected.drop_nulls(subset=required)

    return projected.to_dicts(), before - projected.height


def _frame_from(entities: Sequence[Any], model: type[Any]) -> pl.DataFrame:
    """Build a polars frame from ORM instances.

    Schema inference scans every row: columns such as ``permanent_number`` are null for
    hundreds of historical drivers before the first integer appears, and the default
    100-row sample would infer the wrong type and then fail on append.
    """
    if not entities:
        return pl.DataFrame()
    names = [column.key for column in sa_inspect(model).columns]
    return pl.DataFrame(
        [{name: getattr(entity, name) for name in names} for entity in entities],
        infer_schema_length=None,
    )


def _driver_name_lookup(uow: UnitOfWork) -> dict[str, str]:
    """Map each driver's normalised full name to their canonical id."""
    from f1cp.data.identity import normalise_name

    return {
        normalise_name(f"{driver.given_name}{driver.family_name}"): driver.driver_id
        for driver in uow.drivers.list()
    }


def _seasons_in(bundle: DataBundle) -> list[int]:
    """Seasons represented in a bundle's calendar."""
    races = bundle.get(TableName.RACES)
    if not races.height or "season" not in races.columns:
        return []
    return sorted(races["season"].unique().to_list())


def _seasons_in_results(bundle: DataBundle) -> list[int]:
    """Seasons represented in a bundle's race results."""
    results = bundle.get(TableName.RACE_RESULTS)
    if not results.height or "season" not in results.columns:
        return []
    return sorted(results["season"].unique().to_list())


def _bundle_lineups(bundle: DataBundle) -> pl.DataFrame:
    """Extract ``(season, constructor_id, driver_id)`` rows from a bundle's results."""
    results = bundle.get(TableName.RACE_RESULTS)
    required = {"season", "constructor_id", "driver_id"}
    if not results.height or not required.issubset(results.columns):
        return pl.DataFrame()
    return results.select(["season", "constructor_id", "driver_id"]).unique()


def _stored_lineups(uow: UnitOfWork, seasons: Sequence[int]) -> pl.DataFrame:
    """Load stored ``(season, constructor_id, driver_id)`` rows for the given seasons."""
    if not seasons:
        return pl.DataFrame()

    from sqlalchemy import select

    statement = (
        select(Race.season, RaceResult.constructor_id, RaceResult.driver_id)
        .join(RaceResult, RaceResult.race_id == Race.id)
        .where(Race.season.in_(list(seasons)))
        .distinct()
    )
    rows = uow.session.execute(statement).all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        {
            "season": [row[0] for row in rows],
            "constructor_id": [row[1] for row in rows],
            "driver_id": [row[2] for row in rows],
        },
        schema={"season": pl.Int64, "constructor_id": pl.String, "driver_id": pl.String},
    )


def _summarise_seasons(seasons: Sequence[int]) -> str | None:
    """Render a season list compactly for the audit record."""
    if not seasons:
        return None
    if len(seasons) == 1:
        return str(seasons[0])
    return f"{min(seasons)}-{max(seasons)}"
