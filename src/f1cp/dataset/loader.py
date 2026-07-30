"""Load curated Formula 1 data out of the database as :class:`~f1cp.features.SourceFrames`.

This is the only module that lets the (impure, database-backed) storage layer touch the
(pure, network-free) feature layer. Every table is projected onto the column names and
types the feature blocks expect, and race-scoped tables are joined back to
``(season, round)`` so downstream code never has to know about the surrogate ``race_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from sqlalchemy import func, select

from f1cp.db.models import (
    Circuit,
    Constructor,
    ConstructorStanding,
    DataIngestion,
    Driver,
    DriverStanding,
    QualifyingResult,
    Race,
    RaceResult,
    SprintResult,
)
from f1cp.features.context import SourceFrames

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.engine import Result

    from f1cp.db.uow import UnitOfWork

__all__ = ["load_source_frames", "source_data_revision"]

_SOURCE_TABLES = (
    Race,
    RaceResult,
    SprintResult,
    QualifyingResult,
    DriverStanding,
    ConstructorStanding,
    Driver,
    Constructor,
    Circuit,
)


def _limit_to_season[SelectT: Select[Any]](statement: SelectT, max_season: int | None) -> SelectT:
    """Restrict a race-joined statement to seasons at or below ``max_season``.

    Args:
        statement: A statement already joined to :class:`~f1cp.db.models.Race`.
        max_season: Upper bound, inclusive, or ``None`` for no restriction.

    Returns:
        The statement, filtered when ``max_season`` is given.
    """
    if max_season is None:
        return statement
    return statement.where(Race.season <= max_season)


def load_source_frames(uow: UnitOfWork, *, max_season: int | None = None) -> SourceFrames:
    """Read every curated table needed to build the feature panel.

    Args:
        uow: An open :class:`~f1cp.db.uow.UnitOfWork`.
        max_season: Restrict the race calendar (and everything joined to it) to seasons at
            or below this year. ``None`` loads every stored season.

    Returns:
        A populated :class:`~f1cp.features.context.SourceFrames`.
    """
    races = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    Race.name,
                    Race.date,
                    Race.circuit_id,
                    Race.has_sprint,
                ),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "name": pl.String,
            "date": pl.Date,
            "circuit_id": pl.String,
            "has_sprint": pl.Boolean,
        },
    )

    race_results = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    RaceResult.driver_id,
                    RaceResult.constructor_id,
                    RaceResult.grid,
                    RaceResult.position,
                    RaceResult.position_order,
                    RaceResult.points,
                    RaceResult.laps,
                    RaceResult.status,
                    RaceResult.finished,
                ).join(Race, Race.id == RaceResult.race_id),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "grid": pl.Int64,
            "position": pl.Int64,
            "position_order": pl.Int64,
            "points": pl.Float64,
            "laps": pl.Int64,
            "status": pl.String,
            "finished": pl.Boolean,
        },
    )

    sprint_results = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    SprintResult.driver_id,
                    SprintResult.constructor_id,
                    SprintResult.position,
                    SprintResult.points,
                ).join(Race, Race.id == SprintResult.race_id),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "position": pl.Int64,
            "points": pl.Float64,
        },
    )

    qualifying = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    QualifyingResult.driver_id,
                    QualifyingResult.constructor_id,
                    QualifyingResult.position,
                ).join(Race, Race.id == QualifyingResult.race_id),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "position": pl.Int64,
        },
    )

    driver_standings = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    DriverStanding.driver_id,
                    DriverStanding.constructor_id,
                    DriverStanding.points,
                    DriverStanding.position,
                    DriverStanding.wins,
                ).join(Race, Race.id == DriverStanding.race_id),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "driver_id": pl.String,
            "constructor_id": pl.String,
            "points": pl.Float64,
            "position": pl.Int64,
            "wins": pl.Int64,
        },
    )

    constructor_standings = _to_frame(
        uow.session.execute(
            _limit_to_season(
                select(
                    Race.season,
                    Race.round,
                    ConstructorStanding.constructor_id,
                    ConstructorStanding.points,
                    ConstructorStanding.position,
                    ConstructorStanding.wins,
                ).join(Race, Race.id == ConstructorStanding.race_id),
                max_season,
            )
        ),
        schema={
            "season": pl.Int64,
            "round": pl.Int64,
            "constructor_id": pl.String,
            "points": pl.Float64,
            "position": pl.Int64,
            "wins": pl.Int64,
        },
    )

    drivers = _to_frame(
        uow.session.execute(
            select(
                Driver.driver_id,
                Driver.code,
                Driver.permanent_number,
                Driver.given_name,
                Driver.family_name,
                Driver.date_of_birth,
                Driver.nationality,
            )
        ),
        schema={
            "driver_id": pl.String,
            "code": pl.String,
            "permanent_number": pl.Int64,
            "given_name": pl.String,
            "family_name": pl.String,
            "date_of_birth": pl.Date,
            "nationality": pl.String,
        },
    )

    constructors = _to_frame(
        uow.session.execute(
            select(Constructor.constructor_id, Constructor.name, Constructor.nationality)
        ),
        schema={"constructor_id": pl.String, "name": pl.String, "nationality": pl.String},
    )

    circuits = _to_frame(
        uow.session.execute(
            select(
                Circuit.circuit_id,
                Circuit.name,
                Circuit.locality,
                Circuit.country,
                Circuit.latitude,
                Circuit.longitude,
            )
        ),
        schema={
            "circuit_id": pl.String,
            "name": pl.String,
            "locality": pl.String,
            "country": pl.String,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
        },
    )

    return SourceFrames(
        races=races,
        race_results=race_results,
        sprint_results=sprint_results,
        qualifying=qualifying,
        driver_standings=driver_standings,
        constructor_standings=constructor_standings,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
    )


def source_data_revision(uow: UnitOfWork) -> tuple[int, ...]:
    """A cheap fingerprint of everything :func:`load_source_frames` reads.

    Lets a cache key on the source data without rebuilding the panel to find out whether it
    changed: one round trip over the same nine tables, ~1.5ms against the real database.

    Two signals, because neither alone is enough -- a row count per table catches inserts
    (including fixtures writing directly, bypassing ingestion), and the highest
    ``data_ingestions`` id catches ``update-data --refresh`` overwriting rows in place and
    leaving every count unchanged. It lives beside :func:`load_source_frames` because the
    two must list the same tables.

    Args:
        uow: An open :class:`~f1cp.db.uow.UnitOfWork`.

    Returns:
        An opaque, hashable token. Equal tokens mean the source data has not changed;
        never interpret the individual values.
    """
    counts = [select(func.count()).select_from(table).scalar_subquery() for table in _SOURCE_TABLES]
    latest_ingestion = select(func.coalesce(func.max(DataIngestion.id), 0)).scalar_subquery()
    row = uow.session.execute(select(*counts, latest_ingestion)).one()
    return tuple(int(value) for value in row)


def _to_frame(result: Result[Any], schema: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    """Materialise a SQLAlchemy result into a polars frame with an explicit schema.

    An explicit schema is required here: several columns (``date_of_birth``,
    ``permanent_number``...) are null for long stretches of Formula 1's history, and
    polars' schema inference would otherwise guess the wrong type from a null-heavy sample
    and fail on append.

    Args:
        result: A SQLAlchemy result from ``session.execute(...)``.
        schema: Column name to polars dtype mapping, in the order the query selects them.

    Returns:
        A frame with exactly the given schema.
    """
    columns = list(schema.keys())
    rows = [dict(zip(columns, row, strict=True)) for row in result]
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)
