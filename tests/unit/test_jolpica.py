"""Tests for the Jolpica provider's parsing, against canned API payloads.

The scenario is the one that broke the incremental update after the 2026 Brazilian Grand
Prix: Ryo Hirakawa scored a point as a Haas stand-in earlier in the season, so he is in
every later round's driver standings, but he did not race in Brazil, so the round's
results carry no driver record for him.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest

from f1cp.config.settings import JolpicaProviderConfig
from f1cp.data.identity import IdentityResolver
from f1cp.data.providers.jolpica import JolpicaProvider
from f1cp.domain.enums import TableName

pytestmark = pytest.mark.unit

_SEASON = 2026
_ROUND = 20

_NORRIS = {
    "driverId": "norris",
    "code": "NOR",
    "permanentNumber": "4",
    "givenName": "Lando",
    "familyName": "Norris",
    "dateOfBirth": "1999-11-13",
    "nationality": "British",
}
# Only in the standings: scored earlier in the season, did not race this round.
_HIRAKAWA = {
    "driverId": "ryo_hirakawa",
    "code": "HIR",
    "permanentNumber": "50",
    "givenName": "Ryo",
    "familyName": "Hirakawa",
    "dateOfBirth": "1994-03-07",
    "nationality": "Japanese",
}
_MCLAREN = {"constructorId": "mclaren", "name": "McLaren", "nationality": "British"}
_HAAS = {"constructorId": "haas", "name": "Haas F1 Team", "nationality": "American"}

_RACE = {
    "season": str(_SEASON),
    "round": str(_ROUND),
    "raceName": "Brazilian Grand Prix",
    "date": "2026-11-08",
    "Circuit": {
        "circuitId": "interlagos",
        "circuitName": "Autódromo José Carlos Pace",
        "Location": {"lat": "-23.7", "long": "-46.7", "locality": "São Paulo", "country": "Brazil"},
    },
}


def _envelope(**body: Any) -> dict[str, Any]:
    return {"MRData": {"total": "1", **body}}


def _races(**race_body: Any) -> dict[str, Any]:
    return _envelope(RaceTable={"Races": [{**_RACE, **race_body}]})


class _FakeClient:
    """Serves canned payloads by endpoint, in place of the HTTP client."""

    def get_json(self, path: str, **_: Any) -> dict[str, Any]:
        endpoint = path.rstrip("/").rsplit("/", 1)[-1]
        if endpoint == "races":
            return _races()
        if endpoint == "results":
            return _races(
                Results=[
                    {
                        "position": "1",
                        "positionText": "1",
                        "points": "25",
                        "grid": "1",
                        "laps": "71",
                        "status": "Finished",
                        "Driver": _NORRIS,
                        "Constructor": _MCLAREN,
                    }
                ]
            )
        if endpoint == "driverstandings":
            return _envelope(
                StandingsTable={
                    "StandingsLists": [
                        {
                            "DriverStandings": [
                                {
                                    "position": "1",
                                    "points": "300",
                                    "wins": "5",
                                    "Driver": _NORRIS,
                                    "Constructors": [_MCLAREN],
                                },
                                {
                                    "position": "21",
                                    "points": "1",
                                    "wins": "0",
                                    "Driver": _HIRAKAWA,
                                    "Constructors": [_HAAS],
                                },
                            ]
                        }
                    ]
                }
            )
        if endpoint == "constructorstandings":
            return _envelope(
                StandingsTable={
                    "StandingsLists": [
                        {
                            "ConstructorStandings": [
                                {"position": "1", "points": "500", "wins": "8", "Constructor": c}
                                for c in (_MCLAREN, _HAAS)
                            ]
                        }
                    ]
                }
            )
        # Qualifying, sprints and pit stops are irrelevant to this scenario.
        return _envelope(total="0", RaceTable={"Races": []})

    def close(self) -> None:
        """Nothing to release."""


@pytest.fixture
def provider() -> JolpicaProvider:
    jolpica = JolpicaProvider(JolpicaProviderConfig())
    jolpica._client = _FakeClient()  # type: ignore[assignment]
    return jolpica


def test_drivers_only_in_the_standings_are_still_emitted(provider: JolpicaProvider) -> None:
    bundle = provider.fetch_season(_SEASON, rounds=[_ROUND])

    drivers = bundle.get(TableName.DRIVERS)
    assert set(drivers["driver_id"]) == {"norris", "ryo_hirakawa"}
    hirakawa = drivers.filter(pl.col("driver_id") == "ryo_hirakawa")
    assert hirakawa["date_of_birth"].item() == dt.date(1994, 3, 7)

    constructors = bundle.get(TableName.CONSTRUCTORS)
    assert set(constructors["constructor_id"]) == {"mclaren", "haas"}


def test_standings_only_driver_resolves_to_the_stored_identity(provider: JolpicaProvider) -> None:
    """The stand-in's standings row must carry the canonical id, or its FK insert fails."""
    bundle = provider.fetch_season(_SEASON, rounds=[_ROUND])
    canonical_drivers = pl.DataFrame(
        {
            "driver_id": ["lando-norris", "ryo-hirakawa"],
            "given_name": ["Lando", "Ryo"],
            "family_name": ["Norris", "Hirakawa"],
            "date_of_birth": [dt.date(1999, 11, 13), dt.date(1994, 3, 7)],
        }
    )

    resolver = IdentityResolver.build(
        canonical_drivers=canonical_drivers,
        canonical_constructors=pl.DataFrame({"constructor_id": ["mclaren", "haas"]}),
        canonical_circuits=pl.DataFrame({"circuit_id": ["interlagos"]}),
        foreign_drivers=bundle.get(TableName.DRIVERS),
        foreign_constructors=bundle.get(TableName.CONSTRUCTORS),
        foreign_circuits=bundle.get(TableName.CIRCUITS),
    )
    resolved = resolver.apply(bundle)

    standings = resolved.get(TableName.DRIVER_STANDINGS)
    assert set(standings["driver_id"]) == {"lando-norris", "ryo-hirakawa"}
    assert not resolver.generated
