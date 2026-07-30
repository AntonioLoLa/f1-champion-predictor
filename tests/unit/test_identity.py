"""Tests for cross-provider identifier reconciliation.

The scenarios here are the real failures observed while integrating f1db with Jolpica, not
invented ones.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from f1cp.data.identity import IdentityResolver, normalise_name, slugify
from f1cp.data.providers.base import DataBundle
from f1cp.domain.enums import TableName

pytestmark = pytest.mark.unit


class TestNormalisation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Nico Hülkenberg", "nicohulkenberg"),
            ("Sergio Pérez", "sergioperez"),
            ("RB F1 Team", "rbf1team"),
            ("red_bull", "redbull"),
            ("red-bull", "redbull"),
            (None, ""),
        ],
    )
    def test_normalise_name(self, value: str | None, expected: str) -> None:
        assert normalise_name(value) == expected

    def test_slugify_matches_the_f1db_convention(self) -> None:
        assert slugify("Kimi", "Antonelli") == "kimi-antonelli"
        assert slugify("Lewis", "Hamilton") == "lewis-hamilton"


CANONICAL_DRIVERS = pl.DataFrame(
    {
        "driver_id": ["lewis-hamilton", "max-verstappen", "kimi-antonelli", "duncan-hamilton"],
        "given_name": ["Lewis", "Max", "Kimi", "Duncan"],
        "family_name": ["Hamilton", "Verstappen", "Antonelli", "Hamilton"],
        "date_of_birth": [
            dt.date(1985, 1, 7),
            dt.date(1997, 9, 30),
            dt.date(2006, 8, 25),
            dt.date(1920, 4, 30),
        ],
    }
)

# Ergast identifiers: surnames, sometimes disambiguated, sometimes underscored.
FOREIGN_DRIVERS = pl.DataFrame(
    {
        "driver_id": ["hamilton", "max_verstappen", "antonelli"],
        "given_name": ["Lewis", "Max", "Andrea Kimi"],
        "family_name": ["Hamilton", "Verstappen", "Antonelli"],
        "date_of_birth": [dt.date(1985, 1, 7), dt.date(1997, 9, 30), dt.date(2006, 8, 25)],
    }
)


class TestDriverResolution:
    def test_maps_ergast_surnames_onto_f1db_slugs(self) -> None:
        resolver = _build(foreign_drivers=FOREIGN_DRIVERS)
        assert resolver.drivers == {
            "hamilton": "lewis-hamilton",
            "max_verstappen": "max-verstappen",
            "antonelli": "kimi-antonelli",
        }

    def test_disambiguates_shared_surnames_by_date_of_birth(self) -> None:
        """Two Hamiltons exist; only the date of birth separates them."""
        resolver = _build(foreign_drivers=FOREIGN_DRIVERS)
        assert resolver.drivers["hamilton"] == "lewis-hamilton"

    def test_mints_a_slug_for_a_driver_f1db_has_not_published_yet(self) -> None:
        rookie = pl.DataFrame(
            {
                "driver_id": ["newguy"],
                "given_name": ["New"],
                "family_name": ["Guy"],
                "date_of_birth": [dt.date(2007, 1, 1)],
            }
        )
        resolver = _build(foreign_drivers=rookie)
        assert resolver.drivers["newguy"] == "new-guy"
        assert resolver.generated == ["driver:newguy->new-guy"]


class TestConstructorResolution:
    """The `rb` case: Jolpica's identifier collides with a different f1db entity."""

    CANONICAL_CONSTRUCTORS = pl.DataFrame(
        {
            "constructor_id": ["racing-bulls", "rb", "red-bull"],
            "name": ["Racing Bulls", "RB", "Red Bull"],
        }
    )
    FOREIGN_CONSTRUCTORS = pl.DataFrame(
        {"constructor_id": ["rb", "red_bull"], "name": ["RB F1 Team", "Red Bull"]}
    )

    def test_identifier_collision_is_resolved_by_the_season_lineup(self) -> None:
        """Jolpica's `rb` is f1db's `racing-bulls`, despite an f1db `rb` also existing.

        Neither the identifier ("rb" matches the wrong team) nor the name
        ("RB F1 Team" != "Racing Bulls") disambiguates it. The drivers do.
        """
        canonical_lineups = pl.DataFrame(
            {
                "season": [2026, 2026, 2026, 2026],
                "constructor_id": ["racing-bulls", "racing-bulls", "red-bull", "red-bull"],
                "driver_id": ["liam-lawson", "arvid-lindblad", "max-verstappen", "isack-hadjar"],
            }
        )
        foreign_lineups = pl.DataFrame(
            {
                "season": [2026, 2026, 2026, 2026],
                "constructor_id": ["rb", "rb", "red_bull", "red_bull"],
                "driver_id": ["lawson", "lindblad", "max_verstappen", "hadjar"],
            }
        )
        drivers = pl.DataFrame(
            {
                "driver_id": ["lawson", "lindblad", "max_verstappen", "hadjar"],
                "given_name": ["Liam", "Arvid", "Max", "Isack"],
                "family_name": ["Lawson", "Lindblad", "Verstappen", "Hadjar"],
                "date_of_birth": [
                    dt.date(2002, 2, 11),
                    dt.date(2007, 8, 8),
                    dt.date(1997, 9, 30),
                    dt.date(2004, 9, 28),
                ],
            }
        )
        canonical_drivers = pl.DataFrame(
            {
                "driver_id": ["liam-lawson", "arvid-lindblad", "max-verstappen", "isack-hadjar"],
                "given_name": ["Liam", "Arvid", "Max", "Isack"],
                "family_name": ["Lawson", "Lindblad", "Verstappen", "Hadjar"],
                "date_of_birth": [
                    dt.date(2002, 2, 11),
                    dt.date(2007, 8, 8),
                    dt.date(1997, 9, 30),
                    dt.date(2004, 9, 28),
                ],
            }
        )

        resolver = IdentityResolver.build(
            canonical_drivers=canonical_drivers,
            canonical_constructors=self.CANONICAL_CONSTRUCTORS,
            canonical_circuits=pl.DataFrame(),
            foreign_drivers=drivers,
            foreign_constructors=self.FOREIGN_CONSTRUCTORS,
            foreign_circuits=pl.DataFrame(),
            canonical_lineups=canonical_lineups,
            foreign_lineups=foreign_lineups,
        )

        assert resolver.constructors["rb"] == "racing-bulls"
        assert resolver.constructors["red_bull"] == "red-bull"

    def test_without_lineups_the_identifier_collision_would_win(self) -> None:
        """Documents why the line-up stage exists: name matching alone gets this wrong."""
        resolver = IdentityResolver.build(
            canonical_drivers=CANONICAL_DRIVERS,
            canonical_constructors=self.CANONICAL_CONSTRUCTORS,
            canonical_circuits=pl.DataFrame(),
            foreign_drivers=pl.DataFrame(),
            foreign_constructors=self.FOREIGN_CONSTRUCTORS,
            foreign_circuits=pl.DataFrame(),
        )
        assert resolver.constructors["rb"] == "rb"


class TestApplication:
    def test_rewrites_every_identifier_column(self) -> None:
        resolver = _build(foreign_drivers=FOREIGN_DRIVERS)
        bundle = DataBundle()
        bundle.set(
            TableName.RACE_RESULTS,
            pl.DataFrame(
                {
                    "season": [2026, 2026],
                    "round": [1, 1],
                    "driver_id": ["hamilton", "antonelli"],
                    "points": [18.0, 25.0],
                }
            ),
        )
        bundle.set(
            TableName.DRIVER_STANDINGS,
            pl.DataFrame({"season": [2026], "round": [1], "driver_id": ["hamilton"]}),
        )

        resolver.apply(bundle)

        assert bundle.get(TableName.RACE_RESULTS)["driver_id"].to_list() == [
            "lewis-hamilton",
            "kimi-antonelli",
        ]
        assert bundle.get(TableName.DRIVER_STANDINGS)["driver_id"].to_list() == ["lewis-hamilton"]

    def test_leaves_unrelated_columns_alone(self) -> None:
        resolver = _build(foreign_drivers=FOREIGN_DRIVERS)
        bundle = DataBundle()
        bundle.set(TableName.RACES, pl.DataFrame({"season": [2026], "round": [1]}))
        resolver.apply(bundle)
        assert bundle.get(TableName.RACES).columns == ["season", "round"]


def _build(*, foreign_drivers: pl.DataFrame) -> IdentityResolver:
    return IdentityResolver.build(
        canonical_drivers=CANONICAL_DRIVERS,
        canonical_constructors=pl.DataFrame(),
        canonical_circuits=pl.DataFrame(),
        foreign_drivers=foreign_drivers,
        foreign_constructors=pl.DataFrame(),
        foreign_circuits=pl.DataFrame(),
    )
