"""Which rounds an incremental update asks the provider for.

Uses the synthetic championship, where 2021 is stored with its full calendar but only its
first two rounds run. Every synthetic date is in the past, so every unplayed 2021 round
counts as due.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from f1cp.data.ingest import DataIngestor
from f1cp.data.providers.base import DataBundle, SeasonProvider
from f1cp.db.session import Database
from f1cp.domain.enums import ProviderName
from tests.integration.test_dataset_builder import _seed

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


class _RecordingProvider(SeasonProvider):
    """Records each ``fetch_season`` call and returns nothing, as an empty API would."""

    name: ClassVar[ProviderName] = ProviderName.JOLPICA

    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int] | None]] = []

    def available_seasons(self) -> Sequence[int]:
        return []

    def fetch_season(self, season: int, *, rounds: Iterable[int] | None = None) -> DataBundle:
        self.calls.append((season, None if rounds is None else list(rounds)))
        return DataBundle()


class _Registry:
    def __init__(self, provider: SeasonProvider) -> None:
        self._provider = provider

    def season(self, name: str | None = None) -> SeasonProvider:
        return self._provider


@pytest.fixture
def setup(settings: Settings) -> tuple[DataIngestor, _RecordingProvider]:
    database = Database(settings.database)
    database.create_all()
    _seed(database)
    provider = _RecordingProvider()
    registry: Any = _Registry(provider)
    return DataIngestor(settings, database, registry), provider


def test_a_season_never_stored_is_fetched_whole(
    setup: tuple[DataIngestor, _RecordingProvider],
) -> None:
    """The new-year case: without this, the sync never ingests a season past the bootstrap."""
    ingestor, provider = setup
    ingestor.incremental(seasons=[2022])
    assert provider.calls == [(2022, None)]


def test_a_known_season_fetches_only_its_unplayed_past_rounds(
    setup: tuple[DataIngestor, _RecordingProvider],
) -> None:
    ingestor, provider = setup
    ingestor.incremental(seasons=[2021])
    assert len(provider.calls) == 1
    season, rounds = provider.calls[0]
    assert season == 2021
    assert rounds is not None
    assert rounds
    assert 1 not in rounds
    assert 2 not in rounds


def test_a_complete_season_makes_no_request(
    setup: tuple[DataIngestor, _RecordingProvider],
) -> None:
    ingestor, provider = setup
    ingestor.incremental(seasons=[2020])
    assert provider.calls == []
