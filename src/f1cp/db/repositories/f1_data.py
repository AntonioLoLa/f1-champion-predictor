"""Repositories for curated Formula 1 data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

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
from f1cp.db.repositories.base import SqlAlchemyRepository, bulk_upsert

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


__all__ = [
    "CircuitRepository",
    "ConstructorRepository",
    "DriverRepository",
    "RaceRepository",
    "ResultRepository",
    "StandingsRepository",
]


class CircuitRepository(SqlAlchemyRepository[Circuit]):
    """Circuit reference data."""

    model = Circuit

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update circuits keyed by ``circuit_id``."""
        return bulk_upsert(self.session, Circuit, rows, index_elements=["circuit_id"])


class DriverRepository(SqlAlchemyRepository[Driver]):
    """Driver reference data."""

    model = Driver

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update drivers keyed by ``driver_id``."""
        return bulk_upsert(self.session, Driver, rows, index_elements=["driver_id"])

    def get_by_ids(self, driver_ids: Sequence[str]) -> Sequence[Driver]:
        """Fetch several drivers at once.

        Args:
            driver_ids: Identifiers to look up.

        Returns:
            The matching drivers, in arbitrary order.
        """
        if not driver_ids:
            return []
        statement = select(Driver).where(Driver.driver_id.in_(driver_ids))
        return self.session.execute(statement).scalars().all()


class ConstructorRepository(SqlAlchemyRepository[Constructor]):
    """Constructor reference data."""

    model = Constructor

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update constructors keyed by ``constructor_id``."""
        return bulk_upsert(self.session, Constructor, rows, index_elements=["constructor_id"])


class RaceRepository(SqlAlchemyRepository[Race]):
    """Race calendar."""

    model = Race

    def upsert_many(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update races keyed by ``(season, round)``."""
        return bulk_upsert(self.session, Race, rows, index_elements=["season", "round"])

    def id_map(self, seasons: Sequence[int] | None = None) -> dict[tuple[int, int], int]:
        """Map ``(season, round)`` to the surrogate race id.

        Results arrive from providers keyed by season and round; this resolves them to
        foreign keys in one query instead of one lookup per row.

        Args:
            seasons: Restrict the map to these seasons, or ``None`` for all.

        Returns:
            Mapping of ``(season, round)`` to race id.
        """
        statement = select(Race.season, Race.round, Race.id)
        if seasons:
            statement = statement.where(Race.season.in_(seasons))
        return {
            (season, rnd): race_id for season, rnd, race_id in self.session.execute(statement).all()
        }

    def seasons(self) -> Sequence[int]:
        """Every season present in the database, ascending."""
        statement = select(Race.season).distinct().order_by(Race.season)
        return self.session.execute(statement).scalars().all()

    def max_round(self, season: int) -> int | None:
        """Highest stored round for a season, or ``None`` if the season is absent.

        Args:
            season: Championship year.

        Returns:
            The latest round number stored for that season.
        """
        statement = select(func.max(Race.round)).where(Race.season == season)
        return self.session.execute(statement).scalar_one_or_none()

    def for_season(self, season: int) -> Sequence[Race]:
        """A season's full published calendar, in round order.

        Includes rounds that have not been run yet: the calendar is known in advance, and
        a view of the season is incomplete without the races still to come.

        Args:
            season: Championship year.

        Returns:
            Every stored race for that season, ordered by round.
        """
        statement = select(Race).where(Race.season == season).order_by(Race.round)
        return self.session.execute(statement).scalars().all()

    def remaining(self, season: int, *, after_round: int) -> Sequence[Race]:
        """The season's calendar after one round, in round order.

        The calendar includes rounds that have not been run yet, which is what lets
        title-contention reasoning (:mod:`f1cp.domain.elimination`) know exactly how many
        points -- and which of those rounds carry a sprint -- are still up for grabs.

        Args:
            season: Championship year.
            after_round: Only rounds strictly after this one are returned.

        Returns:
            Remaining races, ascending by round.
        """
        statement = (
            select(Race).where(Race.season == season, Race.round > after_round).order_by(Race.round)
        )
        return self.session.execute(statement).scalars().all()

    def latest_completed(self) -> Race | None:
        """The most recent race that has results stored.

        Used by the API to answer "predict the current state of the championship" without
        the caller having to know which round was last run.

        Returns:
            The latest race with at least one result row, or ``None``.
        """
        statement = (
            select(Race)
            .join(RaceResult, RaceResult.race_id == Race.id)
            .order_by(Race.season.desc(), Race.round.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalars().first()

    def rounds_with_results(self, season: int) -> Sequence[int]:
        """Rounds of a season that already have results stored.

        Args:
            season: Championship year.

        Returns:
            Ascending round numbers.
        """
        statement = (
            select(Race.round)
            .join(RaceResult, RaceResult.race_id == Race.id)
            .where(Race.season == season)
            .distinct()
            .order_by(Race.round)
        )
        return self.session.execute(statement).scalars().all()


class ResultRepository(SqlAlchemyRepository[RaceResult]):
    """Grand Prix, sprint, qualifying and pit-stop results."""

    model = RaceResult

    def upsert_race_results(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update Grand Prix results keyed by ``(race_id, driver_id)``."""
        return bulk_upsert(self.session, RaceResult, rows, index_elements=["race_id", "driver_id"])

    def upsert_sprint_results(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update sprint results keyed by ``(race_id, driver_id)``."""
        return bulk_upsert(
            self.session, SprintResult, rows, index_elements=["race_id", "driver_id"]
        )

    def upsert_qualifying(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update qualifying results keyed by ``(race_id, driver_id)``."""
        return bulk_upsert(
            self.session, QualifyingResult, rows, index_elements=["race_id", "driver_id"]
        )

    def upsert_pit_stops(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update pit stops keyed by ``(race_id, driver_id, stop)``."""
        return bulk_upsert(
            self.session, PitStop, rows, index_elements=["race_id", "driver_id", "stop"]
        )

    def for_race(self, season: int, round_number: int) -> Sequence[RaceResult]:
        """The classification of one Grand Prix, winner first.

        Ordered by ``position_order`` rather than ``position``: a driver who retired has no
        finishing position at all, and ordering by a nullable column would scatter the
        retirements instead of leaving them at the bottom where the classification puts
        them.

        Args:
            season: Championship year.
            round_number: Round within the season.

        Returns:
            Result rows in classified order.
        """
        statement = (
            select(RaceResult)
            .join(Race, Race.id == RaceResult.race_id)
            .where(Race.season == season, Race.round == round_number)
            .order_by(RaceResult.position_order)
        )
        return self.session.execute(statement).scalars().all()

    def pit_stops_for_race(self, season: int, round_number: int) -> Sequence[PitStop]:
        """Every pit stop of one Grand Prix, grid order then stop number.

        Args:
            season: Championship year.
            round_number: Round within the season.

        Returns:
            Pit stop rows. Empty for anything before 2011, which the source data simply
            does not cover, and for rounds not yet run.
        """
        statement = (
            select(PitStop)
            .join(Race, Race.id == PitStop.race_id)
            .where(Race.season == season, Race.round == round_number)
            .order_by(PitStop.driver_id, PitStop.stop)
        )
        return self.session.execute(statement).scalars().all()

    def winners(self, season: int) -> dict[int, tuple[str, str | None]]:
        """Who won each round of a season.

        Args:
            season: Championship year.

        Returns:
            Mapping of round number to ``(driver_id, constructor_id)`` for the winner.
            Rounds that have not been run yet are simply absent.
        """
        statement = (
            select(Race.round, RaceResult.driver_id, RaceResult.constructor_id)
            .join(Race, Race.id == RaceResult.race_id)
            .where(Race.season == season, RaceResult.position == 1)
        )
        return {
            row.round: (row.driver_id, row.constructor_id)
            for row in self.session.execute(statement)
        }


class StandingsRepository(SqlAlchemyRepository[DriverStanding]):
    """Championship standings after each round."""

    model = DriverStanding

    def upsert_driver_standings(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update driver standings keyed by ``(race_id, driver_id)``."""
        return bulk_upsert(
            self.session, DriverStanding, rows, index_elements=["race_id", "driver_id"]
        )

    def upsert_constructor_standings(self, rows: Iterable[dict[str, Any]]) -> int:
        """Insert or update constructor standings keyed by ``(race_id, constructor_id)``."""
        return bulk_upsert(
            self.session,
            ConstructorStanding,
            rows,
            index_elements=["race_id", "constructor_id"],
        )

    def for_round(self, season: int, round_number: int) -> Sequence[DriverStanding]:
        """Every driver's standing after one specific round, best-placed first.

        Args:
            season: Championship year.
            round_number: Round within the season.

        Returns:
            Standing rows ordered by position.
        """
        statement = (
            select(DriverStanding)
            .join(Race, Race.id == DriverStanding.race_id)
            .where(Race.season == season, Race.round == round_number)
            .order_by(DriverStanding.position)
        )
        return self.session.execute(statement).scalars().all()

    def constructors_for_round(
        self, season: int, round_number: int
    ) -> Sequence[ConstructorStanding]:
        """Every constructor's standing after one specific round, best-placed first.

        Args:
            season: Championship year.
            round_number: Round within the season.

        Returns:
            Constructor standing rows ordered by position.
        """
        statement = (
            select(ConstructorStanding)
            .join(Race, Race.id == ConstructorStanding.race_id)
            .where(Race.season == season, Race.round == round_number)
            .order_by(ConstructorStanding.position)
        )
        return self.session.execute(statement).scalars().all()

    def champion_by_season(self) -> dict[int, str]:
        """Map each completed season to the driver who won the championship.

        The champion is the leader of the standings after the final round of the season,
        which is exactly the label the models are trained to predict. A season only gets an
        entry once its full calendar has results: otherwise the leader after the last
        *ingested* round of an in-progress season (say, round 11 of 22) would be labelled
        the champion, which is simply wrong until the season actually finishes.

        Returns:
            Mapping of season to winning ``driver_id``, for completed seasons only.
        """
        calendar_final_round = (
            select(Race.season, func.max(Race.round).label("calendar_final_round"))
            .group_by(Race.season)
            .subquery()
        )
        results_final_round = (
            select(Race.season, func.max(Race.round).label("results_final_round"))
            .join(DriverStanding, DriverStanding.race_id == Race.id)
            .group_by(Race.season)
            .subquery()
        )
        statement = (
            select(Race.season, DriverStanding.driver_id)
            .join(DriverStanding, DriverStanding.race_id == Race.id)
            .join(calendar_final_round, calendar_final_round.c.season == Race.season)
            .join(results_final_round, results_final_round.c.season == Race.season)
            .where(
                Race.round == calendar_final_round.c.calendar_final_round,
                calendar_final_round.c.calendar_final_round
                == results_final_round.c.results_final_round,
                DriverStanding.position == 1,
            )
        )
        return dict(self.session.execute(statement).all())  # type: ignore[arg-type]
