"""Reconciliation of entity identifiers across providers.

f1db and Jolpica use incompatible vocabularies for the same real-world entities:

============  ==========================  ==========================
Entity        f1db                        Jolpica (Ergast)
============  ==========================  ==========================
Driver        ``lewis-hamilton``          ``hamilton``
Driver        ``max-verstappen``          ``max_verstappen``
Constructor   ``kick-sauber``             ``sauber``
============  ==========================  ==========================

Writing both vocabularies into one database would create two rows for Lewis Hamilton and
silently destroy every career-level feature. This module maps one vocabulary onto the
other so that exactly one identifier survives.

**f1db is canonical.** Its identifiers are full-name slugs, which are stable and
unambiguous, whereas Ergast identifiers are ad-hoc surnames that collide (``hamilton``
covers Lewis but needs disambiguation for Duncan).

Matching drivers on ``(date of birth, surname)`` was measured against the full history:
782 of the 792 drivers who have ever started a Grand Prix resolve correctly (98.7%). All
ten failures are one-off Indianapolis 500 entrants from the 1950s -- 46 starts in total,
none after 1970, none inside the modelling window.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from f1cp.domain.enums import TableName
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from f1cp.data.providers.base import DataBundle

__all__ = ["IdentityResolver", "normalise_name", "slugify"]

logger = get_logger(__name__)

# Columns that hold each kind of identifier, per canonical table.
_DRIVER_ID_TABLES = (
    TableName.RACE_RESULTS,
    TableName.SPRINT_RESULTS,
    TableName.QUALIFYING_RESULTS,
    TableName.DRIVER_STANDINGS,
    TableName.PIT_STOPS,
    TableName.DRIVERS,
)
_CONSTRUCTOR_ID_TABLES = (
    TableName.RACE_RESULTS,
    TableName.SPRINT_RESULTS,
    TableName.QUALIFYING_RESULTS,
    TableName.DRIVER_STANDINGS,
    TableName.CONSTRUCTOR_STANDINGS,
    TableName.CONSTRUCTORS,
)
_CIRCUIT_ID_TABLES = (TableName.RACES, TableName.CIRCUITS)


def normalise_name(value: object) -> str:
    """Reduce a name to a comparable key.

    Strips accents, punctuation, spacing and case, so that ``"Nico Hülkenberg"`` and
    ``"Nico Hulkenberg"`` compare equal.

    Args:
        value: Any value; ``None`` becomes an empty string.

    Returns:
        A lowercase alphanumeric key.
    """
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", without_accents.lower())


def slugify(*parts: object) -> str:
    """Build an f1db-style identifier from name parts.

    Used when a driver debuts before the next f1db release. f1db's convention is a
    hyphenated full-name slug, so generating the same shape means the identifier converges
    once the release catches up.

    Args:
        *parts: Name components, e.g. given name and family name.

    Returns:
        A hyphenated slug such as ``"kimi-antonelli"``.
    """
    cleaned = [normalise_name(part) for part in parts]
    return "-".join(part for part in cleaned if part)


@dataclass(slots=True)
class IdentityResolver:
    """Translates a foreign provider's identifiers into the canonical f1db vocabulary.

    Attributes:
        drivers: Foreign driver id to canonical driver id.
        constructors: Foreign constructor id to canonical constructor id.
        circuits: Foreign circuit id to canonical circuit id.
        generated: Identifiers that had no canonical match and were minted from a name
            slug. Non-empty means a rookie or a new team appeared before f1db published
            them; worth logging, and self-healing at the next bootstrap.
    """

    drivers: dict[str, str] = field(default_factory=dict)
    constructors: dict[str, str] = field(default_factory=dict)
    circuits: dict[str, str] = field(default_factory=dict)
    generated: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        canonical_drivers: pl.DataFrame,
        canonical_constructors: pl.DataFrame,
        canonical_circuits: pl.DataFrame,
        foreign_drivers: pl.DataFrame,
        foreign_constructors: pl.DataFrame,
        foreign_circuits: pl.DataFrame,
        canonical_lineups: pl.DataFrame | None = None,
        foreign_lineups: pl.DataFrame | None = None,
    ) -> IdentityResolver:
        """Build a mapping from foreign identifiers to canonical ones.

        Args:
            canonical_drivers: Canonical ``drivers`` rows already in the database.
            canonical_constructors: Canonical ``constructors`` rows.
            canonical_circuits: Canonical ``circuits`` rows.
            foreign_drivers: Incoming ``drivers`` rows in the foreign vocabulary.
            foreign_constructors: Incoming ``constructors`` rows.
            foreign_circuits: Incoming ``circuits`` rows.
            canonical_lineups: ``(season, constructor_id, driver_id)`` rows already stored.
            foreign_lineups: The same, from the incoming bundle. When both are supplied,
                constructors are matched by shared driver line-up, which survives the team
                renames that defeat name matching.

        Returns:
            A populated resolver.
        """
        resolver = cls()
        resolver._map_drivers(canonical_drivers, foreign_drivers)

        # Drivers resolve reliably on date of birth, so the line-up match runs first and
        # name matching only fills whatever it leaves over.
        if canonical_lineups is not None and foreign_lineups is not None:
            resolver._map_constructors_by_lineup(canonical_lineups, foreign_lineups)

        resolver._map_named(
            canonical_constructors,
            foreign_constructors,
            id_column="constructor_id",
            name_columns=("name",),
            target=resolver.constructors,
        )
        resolver._map_named(
            canonical_circuits,
            foreign_circuits,
            id_column="circuit_id",
            name_columns=("name",),
            target=resolver.circuits,
        )
        return resolver

    def _map_constructors_by_lineup(
        self, canonical: pl.DataFrame, foreign: pl.DataFrame
    ) -> None:
        """Match constructors by the set of drivers they field in a season.

        Name and identifier matching both fail across F1's frequent team renames: Jolpica
        calls the 2026 Faenza team ``rb`` ("RB F1 Team") while f1db calls it
        ``racing-bulls`` ("Racing Bulls"), and f1db separately holds a legacy ``rb``
        entity, so an identifier match silently succeeds against the wrong team.

        A line-up is a far stronger signal: within one season, two constructors that field
        the same drivers are the same constructor.

        Args:
            canonical: ``(season, constructor_id, driver_id)`` rows already stored.
            foreign: The same from the incoming bundle, with driver ids already canonical.
        """
        if not canonical.height or not foreign.height:
            return

        canonical_lineups: dict[tuple[int, str], set[str]] = {}
        for row in canonical.iter_rows(named=True):
            key = (row["season"], row["constructor_id"])
            canonical_lineups.setdefault(key, set()).add(row["driver_id"])

        foreign_lineups: dict[tuple[int, str], set[str]] = {}
        for row in foreign.iter_rows(named=True):
            driver = self.drivers.get(row["driver_id"], row["driver_id"])
            key = (row["season"], row["constructor_id"])
            foreign_lineups.setdefault(key, set()).add(driver)

        for (season, foreign_id), drivers in foreign_lineups.items():
            if foreign_id in self.constructors or not drivers:
                continue

            best_id, best_overlap = None, 0
            for (canonical_season, canonical_id), canonical_drivers in (
                canonical_lineups.items()
            ):
                if canonical_season != season:
                    continue
                overlap = len(drivers & canonical_drivers)
                if overlap > best_overlap:
                    best_id, best_overlap = canonical_id, overlap

            # One shared driver in the same season is coincidence-proof enough: a driver
            # races for at most a couple of teams a year.
            if best_id is not None and best_overlap > 0:
                self.constructors[foreign_id] = best_id

    def _map_drivers(self, canonical: pl.DataFrame, foreign: pl.DataFrame) -> None:
        """Match drivers on date of birth and surname, then on full name."""
        if not foreign.height:
            return

        by_dob_surname: dict[tuple[object, str], str] = {}
        by_full_name: dict[str, str] = {}
        dob_counts: dict[object, list[str]] = {}

        for row in canonical.iter_rows(named=True) if canonical.height else ():
            driver_id = row["driver_id"]
            dob = row.get("date_of_birth")
            surname = normalise_name(row.get("family_name"))
            by_dob_surname[(dob, surname)] = driver_id
            by_full_name[normalise_name(f"{row.get('given_name')}{row.get('family_name')}")] = (
                driver_id
            )
            dob_counts.setdefault(dob, []).append(driver_id)

        for row in foreign.iter_rows(named=True):
            foreign_id = row["driver_id"]
            dob = row.get("date_of_birth")
            surname = normalise_name(row.get("family_name"))
            full_name = normalise_name(f"{row.get('given_name')}{row.get('family_name')}")

            canonical_id = by_dob_surname.get((dob, surname)) or by_full_name.get(full_name)
            if canonical_id is None and dob is not None:
                # A date of birth shared with exactly one known driver is decisive even
                # when the surname was transliterated differently.
                candidates = dob_counts.get(dob, [])
                if len(candidates) == 1:
                    canonical_id = candidates[0]

            if canonical_id is None:
                canonical_id = slugify(row.get("given_name"), row.get("family_name"))
                self.generated.append(f"driver:{foreign_id}->{canonical_id}")

            self.drivers[foreign_id] = canonical_id

    def _map_named(
        self,
        canonical: pl.DataFrame,
        foreign: pl.DataFrame,
        *,
        id_column: str,
        name_columns: tuple[str, ...],
        target: dict[str, str],
    ) -> None:
        """Match entities by normalised name, falling back to an identical identifier."""
        if not foreign.height:
            return

        lookup: dict[str, str] = {}
        known_ids: set[str] = set()
        for row in canonical.iter_rows(named=True) if canonical.height else ():
            entity_id = row[id_column]
            known_ids.add(entity_id)
            lookup.setdefault(normalise_name(entity_id), entity_id)
            for column in name_columns:
                if (value := row.get(column)) is not None:
                    lookup.setdefault(normalise_name(value), entity_id)

        for row in foreign.iter_rows(named=True):
            foreign_id = row[id_column]
            if foreign_id in target:
                # Already resolved by a stronger signal, such as the season line-up match.
                continue
            candidate = None
            if foreign_id in known_ids:
                candidate = foreign_id
            else:
                for column in (*name_columns, id_column):
                    if (value := row.get(column)) is not None:
                        candidate = lookup.get(normalise_name(value))
                        if candidate is not None:
                            break

            if candidate is None:
                candidate = slugify(row.get(name_columns[0]) or foreign_id)
                self.generated.append(f"{id_column}:{foreign_id}->{candidate}")

            target[foreign_id] = candidate


    def apply(self, bundle: DataBundle) -> DataBundle:
        """Rewrite every identifier column of a bundle into the canonical vocabulary.

        Args:
            bundle: Bundle produced by a foreign provider.

        Returns:
            The same bundle, with identifiers translated in place.
        """
        _rewrite(bundle, _DRIVER_ID_TABLES, "driver_id", self.drivers)
        _rewrite(bundle, _CONSTRUCTOR_ID_TABLES, "constructor_id", self.constructors)
        _rewrite(bundle, _CIRCUIT_ID_TABLES, "circuit_id", self.circuits)

        if self.generated:
            logger.info(
                "Minted canonical identifiers for entities not yet known to f1db",
                extra={"count": len(self.generated), "identifiers": self.generated[:20]},
            )
        return bundle

    @property
    def is_empty(self) -> bool:
        """Whether the resolver holds no mappings at all."""
        return not (self.drivers or self.constructors or self.circuits)

    def coverage(self) -> Mapping[str, int]:
        """Number of mapped identifiers per entity type, for logging."""
        return {
            "drivers": len(self.drivers),
            "constructors": len(self.constructors),
            "circuits": len(self.circuits),
            "generated": len(self.generated),
        }


def _rewrite(
    bundle: DataBundle,
    tables: Iterable[TableName],
    column: str,
    mapping: Mapping[str, str],
) -> None:
    """Replace values of ``column`` using ``mapping`` in every listed table."""
    if not mapping:
        return
    for table in tables:
        frame = bundle.tables.get(table)
        if frame is None or not frame.height or column not in frame.columns:
            continue
        bundle.tables[table] = frame.with_columns(
            pl.col(column).replace(mapping).alias(column)
        )
