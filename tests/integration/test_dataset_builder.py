"""Integration test for the full dataset build against a real (SQLite) database.

Seeds the synthetic championship used by the feature unit tests through the actual
repositories -- the same upsert path ``DataIngestor`` uses -- then truncates the second
season to simulate a title race still in progress, and asserts
:func:`f1cp.dataset.builder.build_dataset` labels only the completed season's champion.
This is the regression test for the bug fixed in
:meth:`f1cp.db.repositories.f1_data.StandingsRepository.champion_by_season`: the points
leader of an in-progress season must not be mislabelled as champion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from f1cp.dataset.builder import build_dataset
from f1cp.db.session import Database
from tests.unit.features.synthetic import build_synthetic_sources

if TYPE_CHECKING:
    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


def _seed(database: Database) -> None:
    """Load the synthetic championship, with 2021 truncated to its first two rounds."""
    sources = build_synthetic_sources()
    # Simulate a title fight still in progress: only the opening two rounds of 2021 have
    # been run, even though the full four-round calendar is already known.
    truncated = sources.as_of(2021, 2)

    with database.session_scope() as uow:
        uow.circuits.upsert_many(truncated.circuits.to_dicts())
        uow.drivers.upsert_many(truncated.drivers.to_dicts())
        uow.constructors.upsert_many(truncated.constructors.to_dicts())
        # The full calendar (including unrun 2021 rounds 3-4) is known in advance, exactly
        # as a real season's schedule is published before it is run.
        uow.races.upsert_many(sources.races.to_dicts())
        uow.commit()

        race_ids = uow.races.id_map()

        def with_race_id(frame_rows: list[dict[str, object]]) -> list[dict[str, object]]:
            out = []
            for row in frame_rows:
                season, round_number = row.pop("season"), row.pop("round")
                out.append({**row, "race_id": race_ids[(season, round_number)]})
            return out

        uow.results.upsert_race_results(with_race_id(truncated.race_results.to_dicts()))
        uow.results.upsert_sprint_results(with_race_id(truncated.sprint_results.to_dicts()))
        uow.results.upsert_qualifying(with_race_id(truncated.qualifying.to_dicts()))
        uow.standings.upsert_driver_standings(with_race_id(truncated.driver_standings.to_dicts()))
        uow.standings.upsert_constructor_standings(
            with_race_id(truncated.constructor_standings.to_dicts())
        )


def test_build_dataset_labels_only_the_completed_season(settings: Settings) -> None:
    # The synthetic championship only runs four rounds per season; the default
    # `min_rounds_in_season` filter (five) would otherwise drop it entirely.
    settings = settings.model_copy(
        update={
            "features": settings.features.model_copy(
                update={
                    "filters": settings.features.filters.model_copy(
                        update={"min_rounds_in_season": 1}
                    )
                }
            )
        }
    )

    database = Database(settings.database)
    database.create_all()
    _seed(database)

    with database.session_scope() as uow:
        artifact = build_dataset(uow, settings)

    frame = artifact.frame
    assert frame.filter(frame["season"] == 2021)["champion"].sum() == 0

    completed = frame.filter(frame["season"] == 2020)
    final_round = completed["round"].max()
    champions = completed.filter(
        (completed["round"] == final_round) & (completed["champion"] == 1)
    )["driver_id"].to_list()
    assert len(champions) == 1

    database.dispose()
