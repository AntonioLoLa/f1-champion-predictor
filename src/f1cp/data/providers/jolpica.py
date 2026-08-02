"""Jolpica provider: the Ergast-compatible REST API covering 1950 to the present.

This is the authoritative source for everything needed to reconstruct a championship
round by round. It is also the rate-limited one, so the request budget matters:

===========================  ===========================================================
Fetching a whole season      ~84 requests (results and qualifying paginate at 100 rows,
                             and standings and pit stops are one request *per round*)
Fetching one new round       7 requests
===========================  ===========================================================

At 450 requests/hour a full 1981-2026 backfill would take over five hours, which is why
:class:`~f1cp.data.providers.f1db.F1dbProvider` seeds the history and this provider only
ever fills in the rounds that are missing.

Standings are read from the API rather than recomputed from race points: until 1990 the
championship used "best N results" rules, so a driver's total is not the sum of their
race results.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from f1cp.data.http import JsonHttpClient, RateLimit
from f1cp.data.providers.base import Capability, DataBundle, ProviderError, SeasonProvider
from f1cp.domain.enums import ProviderName, TableName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from f1cp.config.settings import JolpicaProviderConfig
    from f1cp.data.http import ResponseCache

__all__ = ["JolpicaProvider"]

logger = get_logger(__name__)

# Pit stop data only exists from 2011 onwards; asking for earlier seasons wastes requests.
_PIT_STOP_MIN_SEASON = 2011
# Qualifying is recorded from 1994; earlier seasons only have grid positions from results.
_QUALIFYING_MIN_SEASON = 1994
# Sprint races were introduced in 2021.
_SPRINT_MIN_SEASON = 2021

_LAP_TIME_PATTERN = re.compile(r"^(?:(?P<minutes>\d+):)?(?P<seconds>\d+(?:\.\d+)?)$")


def _to_int(value: Any) -> int | None:
    """Parse an API string into an int, returning ``None`` when it is not numeric."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse an API string into a float, falling back to ``default``."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lap_time_to_ms(value: str | None) -> int | None:
    """Convert a lap time such as ``"1:30.031"`` or ``"36.604"`` into milliseconds.

    Args:
        value: The formatted time, or ``None``.

    Returns:
        Milliseconds, or ``None`` if the value is absent or unparseable.
    """
    if not value:
        return None
    match = _LAP_TIME_PATTERN.match(value.strip())
    if match is None:
        return None
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))
    return round((minutes * 60 + seconds) * 1000)


class JolpicaProvider(SeasonProvider):
    """Fetches canonical championship data from the Jolpica API."""

    name: ClassVar[ProviderName] = ProviderName.JOLPICA
    capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.SEASON_FETCH})

    def __init__(
        self,
        config: JolpicaProviderConfig,
        *,
        cache: ResponseCache | None = None,
        immutable_before_season: int = 2026,
    ) -> None:
        self._config = config
        self._immutable_before_season = immutable_before_season
        self._client = JsonHttpClient(
            config.base_url,
            rate_limit=RateLimit(config.requests_per_second, config.requests_per_hour),
            cache=cache,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_factor=config.backoff_factor,
            user_agent=config.user_agent,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()

    @property
    def stats(self) -> dict[str, int]:
        """Request and cache-hit counters, useful for verifying the rate-limit budget."""
        return self._client.stats

    def _is_immutable(self, season: int) -> bool:
        """Completed seasons never change, so their responses can be cached forever."""
        return season < self._immutable_before_season

    def _paginate(self, path: str, season: int) -> Iterator[dict[str, Any]]:
        """Yield each page's ``MRData`` block, following ``total``/``offset``.

        Args:
            path: Endpoint path relative to the base URL.
            season: Season being fetched, used to decide cacheability.

        Yields:
            The ``MRData`` mapping of each page.

        Raises:
            ProviderError: If a response is missing the expected envelope.
        """
        limit = self._config.page_limit
        offset = 0
        immutable = self._is_immutable(season)

        while True:
            payload = self._client.get_json(
                path, params={"limit": limit, "offset": offset}, immutable=immutable
            )
            if not isinstance(payload, dict) or "MRData" not in payload:
                msg = f"Unexpected response envelope from {path!r}"
                raise ProviderError(msg)

            data: dict[str, Any] = payload["MRData"]
            yield data

            total = _to_int(data.get("total")) or 0
            offset += limit
            if offset >= total:
                return

    def _races_from_pages(self, path: str, season: int) -> list[dict[str, Any]]:
        """Collect the ``Races`` entries across every page of a paginated endpoint.

        A single race's nested result list can straddle a page boundary, so entries are
        collected flat and merged by round rather than assumed to arrive whole.

        Args:
            path: Endpoint path.
            season: Season being fetched.

        Returns:
            Race dictionaries, possibly with the same round appearing more than once.
        """
        races: list[dict[str, Any]] = []
        for page in self._paginate(path, season):
            races.extend(page.get("RaceTable", {}).get("Races", []))
        return races

    def available_seasons(self) -> Sequence[int]:
        """Every season the API knows about, ascending.

        Returns:
            Season years.
        """
        seasons: list[int] = []
        for page in self._paginate("seasons/", season=1950):
            for entry in page.get("SeasonTable", {}).get("Seasons", []):
                if (year := _to_int(entry.get("season"))) is not None:
                    seasons.append(year)
        return sorted(seasons)

    def fetch_season(self, season: int, *, rounds: Iterable[int] | None = None) -> DataBundle:
        """Fetch one season, or only selected rounds of it.

        Args:
            season: Championship year.
            rounds: Restrict to these rounds. Passing only the missing rounds is what keeps
                an incremental update to roughly seven requests.

        Returns:
            The season's canonical tables.
        """
        # Fetched once and used for both the calendar and the circuit reference data;
        # a second call would be a cache hit but still needless work.
        race_payloads = self._races_from_pages(f"{season}/races/", season)
        races = self._races_frame(season, race_payloads)
        if races.is_empty():
            logger.warning("No races returned", extra={"season": season})
            return DataBundle()

        all_rounds = races["round"].to_list()
        target_rounds = sorted(set(rounds) & set(all_rounds)) if rounds else all_rounds

        results, drivers, constructors = self._fetch_results(season, target_rounds)
        qualifying = self._fetch_qualifying(season, target_rounds)
        driver_standings, constructor_standings = self._fetch_standings(season, target_rounds)
        pit_stops = self._fetch_pit_stops(season, target_rounds)

        # Sprints are always fetched for the whole season, even during a partial update:
        # `has_sprint` is written for every round of the calendar, so deriving it from a
        # subset would wrongly clear the flag on rounds that were not re-fetched.
        all_sprints, sprint_drivers, sprint_constructors = self._fetch_sprints(season)
        sprint_rounds = set(all_sprints["round"].to_list()) if all_sprints.height else set()
        races = races.with_columns(pl.col("round").is_in(sprint_rounds).alias("has_sprint"))

        sprints = (
            all_sprints.filter(pl.col("round").is_in(target_rounds))
            if all_sprints.height
            else all_sprints
        )

        return DataBundle.from_mapping(
            {
                TableName.CIRCUITS: self._circuits_frame(race_payloads),
                TableName.DRIVERS: _drivers_frame(_concat(drivers, sprint_drivers)),
                TableName.CONSTRUCTORS: _dedupe(_concat(constructors, sprint_constructors)),
                TableName.RACES: races,
                TableName.RACE_RESULTS: results,
                TableName.SPRINT_RESULTS: sprints,
                TableName.QUALIFYING_RESULTS: qualifying,
                TableName.DRIVER_STANDINGS: driver_standings,
                TableName.CONSTRUCTOR_STANDINGS: constructor_standings,
                TableName.PIT_STOPS: pit_stops,
            }
        )

    @staticmethod
    def _races_frame(season: int, race_payloads: list[dict[str, Any]]) -> pl.DataFrame:
        """Build the season calendar from raw race payloads."""
        rows = [
            {
                "season": season,
                "round": _to_int(race.get("round")),
                "name": race.get("raceName", ""),
                "date": race.get("date"),
                "circuit_id": race.get("Circuit", {}).get("circuitId"),
            }
            for race in race_payloads
        ]
        if not rows:
            return pl.DataFrame()
        return (
            pl.DataFrame(rows)
            .with_columns(
                pl.col("date").str.to_date(strict=False),
                pl.col("round").cast(pl.Int64),
            )
            .unique(subset=["season", "round"], keep="first", maintain_order=True)
        )

    @staticmethod
    def _circuits_frame(race_payloads: list[dict[str, Any]]) -> pl.DataFrame:
        """Extract circuit reference data from raw race payloads."""
        rows = []
        for race in race_payloads:
            circuit = race.get("Circuit", {})
            location = circuit.get("Location", {})
            rows.append(
                {
                    "circuit_id": circuit.get("circuitId"),
                    "name": circuit.get("circuitName", ""),
                    "locality": location.get("locality"),
                    "country": location.get("country"),
                    "latitude": _to_float(location.get("lat"), default=float("nan")),
                    "longitude": _to_float(location.get("long"), default=float("nan")),
                }
            )
        return _dedupe(pl.DataFrame(rows)) if rows else pl.DataFrame()

    def _fetch_results(
        self, season: int, rounds: Sequence[int]
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Fetch Grand Prix results plus the driver and constructor records they embed."""
        races = self._fetch_nested(season, rounds, "results", "Results")
        result_rows: list[dict[str, Any]] = []
        driver_rows: list[dict[str, Any]] = []
        constructor_rows: list[dict[str, Any]] = []

        for race, entry in races:
            round_number = _to_int(race.get("round"))
            driver = entry.get("Driver", {})
            constructor = entry.get("Constructor", {})
            position_text = str(entry.get("positionText", ""))
            classified = position_text.isdigit()

            result_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "grid": _to_int(entry.get("grid")),
                    "position": _to_int(entry.get("position")) if classified else None,
                    "position_order": _to_int(entry.get("position")) or 99,
                    "points": _to_float(entry.get("points")),
                    "laps": _to_int(entry.get("laps")),
                    "status": entry.get("status"),
                    "finished": classified,
                    "fastest_lap_rank": _to_int(entry.get("FastestLap", {}).get("rank")),
                }
            )
            driver_rows.append(_driver_row(driver))
            constructor_rows.append(_constructor_row(constructor))

        return (
            pl.DataFrame(result_rows) if result_rows else pl.DataFrame(),
            _dedupe(pl.DataFrame(driver_rows)) if driver_rows else pl.DataFrame(),
            _dedupe(pl.DataFrame(constructor_rows)) if constructor_rows else pl.DataFrame(),
        )

    def _fetch_sprints(self, season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Fetch every sprint result of a season. Empty for seasons before 2021.

        Always season-wide: the caller needs the complete set of sprint rounds to set
        ``has_sprint`` correctly on the calendar.
        """
        if season < _SPRINT_MIN_SEASON:
            return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

        races = self._fetch_nested(season, (), "sprint", "SprintResults")
        rows: list[dict[str, Any]] = []
        driver_rows: list[dict[str, Any]] = []
        constructor_rows: list[dict[str, Any]] = []

        for race, entry in races:
            driver = entry.get("Driver", {})
            constructor = entry.get("Constructor", {})
            position_text = str(entry.get("positionText", ""))
            classified = position_text.isdigit()
            rows.append(
                {
                    "season": season,
                    "round": _to_int(race.get("round")),
                    "driver_id": driver.get("driverId"),
                    "constructor_id": constructor.get("constructorId"),
                    "grid": _to_int(entry.get("grid")),
                    "position": _to_int(entry.get("position")) if classified else None,
                    "position_order": _to_int(entry.get("position")) or 99,
                    "points": _to_float(entry.get("points")),
                    "laps": _to_int(entry.get("laps")),
                    "status": entry.get("status"),
                    "finished": classified,
                }
            )
            driver_rows.append(_driver_row(driver))
            constructor_rows.append(_constructor_row(constructor))

        return (
            pl.DataFrame(rows) if rows else pl.DataFrame(),
            _dedupe(pl.DataFrame(driver_rows)) if driver_rows else pl.DataFrame(),
            _dedupe(pl.DataFrame(constructor_rows)) if constructor_rows else pl.DataFrame(),
        )

    def _fetch_qualifying(self, season: int, rounds: Sequence[int]) -> pl.DataFrame:
        """Fetch qualifying results. Only recorded from 1994 onwards."""
        if season < _QUALIFYING_MIN_SEASON:
            return pl.DataFrame()

        races = self._fetch_nested(season, rounds, "qualifying", "QualifyingResults")
        rows = [
            {
                "season": season,
                "round": _to_int(race.get("round")),
                "driver_id": entry.get("Driver", {}).get("driverId"),
                "constructor_id": entry.get("Constructor", {}).get("constructorId"),
                "position": _to_int(entry.get("position")),
                "q1_ms": _lap_time_to_ms(entry.get("Q1")),
                "q2_ms": _lap_time_to_ms(entry.get("Q2")),
                "q3_ms": _lap_time_to_ms(entry.get("Q3")),
            }
            for race, entry in races
        ]
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def _fetch_nested(
        self,
        season: int,
        rounds: Sequence[int],
        endpoint: str,
        entries_key: str,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Fetch a race-nested endpoint and flatten it into ``(race, entry)`` pairs.

        Fetches the whole season in paginated form when most of it is wanted, and round by
        round when only a few rounds are missing, choosing whichever costs fewer requests.

        Args:
            season: Championship year.
            rounds: Rounds of interest.
            endpoint: Endpoint segment, e.g. ``"results"``.
            entries_key: Key holding the nested list, e.g. ``"Results"``.

        Returns:
            Flattened pairs of race metadata and one nested entry.
        """
        # A season-wide fetch costs ceil(total/100) requests; per-round costs len(rounds).
        # Roughly 20 entries per round means 5 rounds fill a page.
        fetch_per_round = 0 < len(rounds) <= 5

        races: list[dict[str, Any]] = []
        if fetch_per_round:
            for round_number in rounds:
                path = f"{season}/{round_number}/{endpoint}/"
                for page in self._paginate(path, season):
                    races.extend(page.get("RaceTable", {}).get("Races", []))
        else:
            races = self._races_from_pages(f"{season}/{endpoint}/", season)

        wanted = set(rounds)
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for race in races:
            race_round = _to_int(race.get("round"))
            if race_round is None or (wanted and race_round not in wanted):
                continue
            pairs.extend((race, entry) for entry in race.get(entries_key, []))
        return pairs

    def _fetch_standings(
        self, season: int, rounds: Sequence[int]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch driver and constructor standings after each requested round.

        This is the expensive endpoint: standings are only addressable per round, so a full
        season costs two requests per round.
        """
        driver_rows: list[dict[str, Any]] = []
        constructor_rows: list[dict[str, Any]] = []
        immutable = self._is_immutable(season)

        for round_number in rounds:
            payload = self._client.get_json(
                f"{season}/{round_number}/driverstandings/",
                params={"limit": self._config.page_limit},
                immutable=immutable,
            )
            for standings_list in _standings_lists(payload):
                for entry in standings_list.get("DriverStandings", []):
                    constructors = entry.get("Constructors") or [{}]
                    driver_rows.append(
                        {
                            "season": season,
                            "round": round_number,
                            "driver_id": entry.get("Driver", {}).get("driverId"),
                            # A driver can change team mid-season; the last entry is the
                            # team they were driving for at this round.
                            "constructor_id": constructors[-1].get("constructorId"),
                            "points": _to_float(entry.get("points")),
                            "position": _to_int(entry.get("position")),
                            "wins": _to_int(entry.get("wins")) or 0,
                        }
                    )

            payload = self._client.get_json(
                f"{season}/{round_number}/constructorstandings/",
                params={"limit": self._config.page_limit},
                immutable=immutable,
            )
            for standings_list in _standings_lists(payload):
                for entry in standings_list.get("ConstructorStandings", []):
                    constructor_rows.append(
                        {
                            "season": season,
                            "round": round_number,
                            "constructor_id": entry.get("Constructor", {}).get("constructorId"),
                            "points": _to_float(entry.get("points")),
                            "position": _to_int(entry.get("position")),
                            "wins": _to_int(entry.get("wins")) or 0,
                        }
                    )

        return (
            pl.DataFrame(driver_rows) if driver_rows else pl.DataFrame(),
            pl.DataFrame(constructor_rows) if constructor_rows else pl.DataFrame(),
        )

    def _fetch_pit_stops(self, season: int, rounds: Sequence[int]) -> pl.DataFrame:
        """Fetch pit stops. Only available from 2011 onwards."""
        if season < _PIT_STOP_MIN_SEASON:
            return pl.DataFrame()

        rows: list[dict[str, Any]] = []
        for round_number in rounds:
            for page in self._paginate(f"{season}/{round_number}/pitstops/", season):
                for race in page.get("RaceTable", {}).get("Races", []):
                    rows.extend(
                        {
                            "season": season,
                            "round": round_number,
                            "driver_id": stop.get("driverId"),
                            "stop": _to_int(stop.get("stop")),
                            "lap": _to_int(stop.get("lap")),
                            "duration_ms": _lap_time_to_ms(stop.get("duration")),
                        }
                        for stop in race.get("PitStops", [])
                    )
        return pl.DataFrame(rows) if rows else pl.DataFrame()


def _standings_lists(payload: Any) -> list[dict[str, Any]]:
    """Extract the ``StandingsLists`` block from a standings response."""
    if not isinstance(payload, dict):
        return []
    lists = payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists if isinstance(lists, list) else []


def _driver_row(driver: dict[str, Any]) -> dict[str, Any]:
    """Map an API driver object onto the canonical ``drivers`` columns."""
    return {
        "driver_id": driver.get("driverId"),
        "code": driver.get("code"),
        "permanent_number": _to_int(driver.get("permanentNumber")),
        "given_name": driver.get("givenName", ""),
        "family_name": driver.get("familyName", ""),
        # Kept as a string here and cast once, after deduplication.
        "date_of_birth": driver.get("dateOfBirth"),
        "nationality": driver.get("nationality"),
    }


def _concat(*frames: pl.DataFrame) -> pl.DataFrame:
    """Vertically concatenate frames, ignoring empty ones.

    Polars refuses to concatenate a zero-column empty frame with a populated one, which is
    exactly what happens for seasons with no sprint races.

    Args:
        *frames: Frames to combine.

    Returns:
        The concatenation of every non-empty frame, or an empty frame.
    """
    populated = [frame for frame in frames if frame.height]
    if not populated:
        return pl.DataFrame()
    if len(populated) == 1:
        return populated[0]
    return pl.concat(populated, how="diagonal_relaxed")


def _constructor_row(constructor: dict[str, Any]) -> dict[str, Any]:
    """Map an API constructor object onto the canonical ``constructors`` columns."""
    return {
        "constructor_id": constructor.get("constructorId"),
        "name": constructor.get("name", ""),
        "nationality": constructor.get("nationality"),
    }


def _drivers_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate drivers and parse their date of birth.

    Driver records are embedded in every result row, so the same driver arrives dozens of
    times per season; the date is parsed once after collapsing them.

    Args:
        frame: Concatenated raw driver rows.

    Returns:
        One row per driver, with ``date_of_birth`` typed as a date.
    """
    if not frame.height:
        return frame
    return frame.unique(subset=["driver_id"], keep="first", maintain_order=True).with_columns(
        pl.col("date_of_birth").str.to_date(strict=False)
    )


def _dedupe(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop duplicate rows, keeping the first occurrence."""
    return frame.unique(keep="first", maintain_order=True) if frame.height else frame
