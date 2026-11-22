"""The whole pipeline, provider to served prediction, over a synthetic championship.

Every other test stops at a module boundary. This one crosses all of them with the real
code on each side: a provider hands ``DataIngestor`` raw tables, the ingestor writes them
to SQLite, ``build_dataset`` engineers features from that database, ``Trainer`` runs
walk-forward validation and logs to a real MLflow store, ``retraining_due`` reads the
result, and ``PredictionService`` loads the model back out of MLflow and scores the live
round. Then a new round is "published" and the update path runs again: the served
prediction has to move to it without a retrain, which is the deployment's whole contract.

The championship is generated here rather than borrowed from
``tests/unit/features/synthetic.py``: that fixture spans three seasons, too few for a
single walk-forward fold, and it is the leakage test's fixture, so it should not grow to
suit this one. This one has eight seasons, the last still in progress. Each season has
one dominant driver, so the model has real signal to learn and the test can check that
it ranks the right driver first.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pytest

from f1cp.data.ingest import DataIngestor
from f1cp.data.providers.base import DataBundle, SeasonProvider
from f1cp.db.session import Database
from f1cp.domain.enums import ProviderName, TableName
from f1cp.domain.scoring import points_system_for
from f1cp.inference.predictor import PredictionService
from f1cp.training.cadence import retraining_due
from f1cp.training.trainer import Trainer
from tests.integration.test_predictor import _build_dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from f1cp.config.settings import Settings

pytestmark = pytest.mark.e2e

SEASONS = tuple(range(2010, 2018))
LIVE_SEASON = SEASONS[-1]
ROUNDS = 6
DRIVERS = ("ana", "ben", "cai", "dev", "eli", "fay")
TEAM_OF = {"ana": "red", "ben": "red", "cai": "blue", "dev": "blue", "eli": "gold", "fay": "gold"}
CIRCUITS = ("north", "south", "east", "west", "hill", "lake")


def star_of(season: int) -> str:
    """The season's dominant driver, a different one each year."""
    return DRIVERS[season % len(DRIVERS)]


def _finishing_order(season: int, round_number: int) -> list[str]:
    """The star wins three rounds in four and is second otherwise; the rest rotate."""
    star = star_of(season)
    others = [driver for driver in DRIVERS if driver != star]
    shift = (season + round_number) % len(others)
    others = others[shift:] + others[:shift]
    if round_number % 4 == 0:
        return [others[0], star, *others[1:]]
    return [star, *others]


def _race_date(season: int, round_number: int) -> dt.date:
    return dt.date(season, 3, 1) + dt.timedelta(days=14 * (round_number - 1))


def _season_tables(season: int, played: int) -> dict[TableName, list[dict[str, Any]]]:
    """Raw provider tables for one season: the full calendar, results for ``played``."""
    system = points_system_for(season)
    tables: dict[TableName, list[dict[str, Any]]] = {table: [] for table in TableName}
    driver_points = dict.fromkeys(DRIVERS, 0.0)
    driver_wins = dict.fromkeys(DRIVERS, 0)
    team_points = dict.fromkeys(set(TEAM_OF.values()), 0.0)
    team_wins = dict.fromkeys(set(TEAM_OF.values()), 0)

    for round_number in range(1, ROUNDS + 1):
        tables[TableName.RACES].append(
            {
                "season": season,
                "round": round_number,
                "name": f"{CIRCUITS[round_number - 1].title()} Grand Prix",
                "date": _race_date(season, round_number),
                "circuit_id": CIRCUITS[round_number - 1],
                "has_sprint": False,
            }
        )
        if round_number > played:
            continue

        for position, driver in enumerate(_finishing_order(season, round_number), start=1):
            points = float(system.race_points[position - 1])
            team = TEAM_OF[driver]
            driver_points[driver] += points
            team_points[team] += points
            if position == 1:
                driver_wins[driver] += 1
                team_wins[team] += 1
            grid = position
            tables[TableName.RACE_RESULTS].append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver,
                    "constructor_id": team,
                    "grid": grid,
                    "position": position,
                    "position_order": position,
                    "points": points,
                    "laps": 50,
                    "status": "Finished",
                    "finished": True,
                    "fastest_lap_rank": position,
                }
            )
            tables[TableName.QUALIFYING_RESULTS].append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver,
                    "constructor_id": team,
                    "position": grid,
                }
            )

        ranked = sorted(DRIVERS, key=lambda d: (-driver_points[d], d))
        for standing, driver in enumerate(ranked, start=1):
            tables[TableName.DRIVER_STANDINGS].append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver,
                    "constructor_id": TEAM_OF[driver],
                    "points": driver_points[driver],
                    "position": standing,
                    "wins": driver_wins[driver],
                }
            )
        ranked_teams = sorted(team_points, key=lambda t: (-team_points[t], t))
        for standing, team in enumerate(ranked_teams, start=1):
            tables[TableName.CONSTRUCTOR_STANDINGS].append(
                {
                    "season": season,
                    "round": round_number,
                    "constructor_id": team,
                    "points": team_points[team],
                    "position": standing,
                    "wins": team_wins[team],
                }
            )

    tables[TableName.CIRCUITS] = [
        {"circuit_id": circuit, "name": circuit.title(), "country": "Testland"}
        for circuit in CIRCUITS
    ]
    tables[TableName.DRIVERS] = [
        {
            "driver_id": driver,
            "given_name": driver.title(),
            "family_name": "Driver",
            "date_of_birth": dt.date(1990, 1, 1) + dt.timedelta(days=100 * index),
            "nationality": "Testlandish",
        }
        for index, driver in enumerate(DRIVERS)
    ]
    tables[TableName.CONSTRUCTORS] = [
        {"constructor_id": team, "name": team.title(), "nationality": "Testlandish"}
        for team in sorted(set(TEAM_OF.values()))
    ]
    return tables


class _SyntheticProvider(SeasonProvider):
    """Serves the synthetic championship as a provider would, up to a publication point.

    Named as f1db, the canonical vocabulary, so ingestion stores these identifiers as
    they are. Cross-provider identity resolution has its own tests and is not the subject.
    """

    name: ClassVar[ProviderName] = ProviderName.F1DB

    def __init__(self) -> None:
        self.live_rounds_published = 3
        self.calls: list[tuple[int, list[int] | None]] = []

    def available_seasons(self) -> Sequence[int]:
        return SEASONS

    def fetch_season(self, season: int, *, rounds: Iterable[int] | None = None) -> DataBundle:
        wanted = None if rounds is None else sorted(rounds)
        self.calls.append((season, wanted))
        played = self.live_rounds_published if season == LIVE_SEASON else ROUNDS
        tables = _season_tables(season, played)
        if wanted is not None:
            scoped = {
                TableName.RACE_RESULTS,
                TableName.QUALIFYING_RESULTS,
                TableName.DRIVER_STANDINGS,
                TableName.CONSTRUCTOR_STANDINGS,
            }
            for table in scoped:
                tables[table] = [row for row in tables[table] if row["round"] in wanted]
        return DataBundle.from_mapping(
            {table: pl.DataFrame(rows) for table, rows in tables.items() if rows}
        )


class _Registry:
    def __init__(self, provider: SeasonProvider) -> None:
        self._provider = provider

    def season(self, name: str | None = None) -> SeasonProvider:
        return self._provider


def _e2e_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Shrink walk-forward validation and the zoo to fit eight seasons and a test budget."""
    validation = settings.training.validation.model_copy(
        update={
            "strategy": "expanding",
            "min_train_seasons": 3,
            "test_horizon_seasons": 1,
            "calibration_holdout_seasons": 1,
        }
    )
    fast_models = {"logistic_regression", "random_forest"}
    return settings.model_copy(
        update={
            "training": settings.training.model_copy(update={"validation": validation}),
            "model_zoo": {
                name: spec.model_copy(update={"enabled": name in fast_models})
                for name, spec in settings.model_zoo.items()
            },
            "features": settings.features.model_copy(
                update={
                    "filters": settings.features.filters.model_copy(
                        update={"min_rounds_in_season": 1}
                    )
                }
            ),
            "mlflow": settings.mlflow.model_copy(
                update={"enabled": True, "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"}
            ),
        }
    )


def test_provider_to_served_prediction_and_the_next_round(
    settings: Settings, tmp_path: Path
) -> None:
    e2e_settings = _e2e_settings(settings, tmp_path)
    database = Database(e2e_settings.database)
    database.create_all()
    provider = _SyntheticProvider()
    registry: Any = _Registry(provider)
    ingestor = DataIngestor(e2e_settings, database, registry)

    # 1. Ingest. The database starts empty, so every season is unknown and fetched whole.
    report = ingestor.incremental(seasons=SEASONS)
    assert sorted(report.seasons) == list(SEASONS)
    assert all(rounds is None for _, rounds in provider.calls)

    # 2. Seven seasons have closed and nothing is trained: a retrain is due.
    with database.session_scope() as uow:
        assert retraining_due(uow)

    # 3. Build the dataset from the database, then train and register a model.
    _build_dataset(database, e2e_settings)
    result = Trainer(e2e_settings, database).run()
    assert result.best_model_name in {"logistic_regression", "random_forest"}
    with database.session_scope() as uow:
        assert not retraining_due(uow)

    # 4. Serve the live round, with the model loaded back out of MLflow.
    service = PredictionService(e2e_settings)
    with database.session_scope() as uow:
        live = service.predict_round(uow)
    assert (live.season, live.round) == (LIVE_SEASON, 3)
    assert live.model_version_id == result.best_model_version_id
    assert live.is_out_of_sample
    assert {driver.driver_id for driver in live.drivers} == set(DRIVERS)
    assert sum(driver.probability for driver in live.drivers) == pytest.approx(1.0)
    assert live.drivers[0].driver_id == star_of(LIVE_SEASON)

    # 5. A round is published. The update fetches only the rounds now due, and the served
    #    prediction moves to it with the same model: no retrain mid-season.
    provider.live_rounds_published = 4
    provider.calls.clear()
    ingestor.incremental(seasons=[LIVE_SEASON])
    assert provider.calls == [(LIVE_SEASON, [4, 5, 6])]
    with database.session_scope() as uow:
        assert not retraining_due(uow), "a round mid-season adds no label"
        after = service.predict_round(uow)
    assert (after.season, after.round) == (LIVE_SEASON, 4)
    assert after.model_version_id == result.best_model_version_id
    assert sum(driver.probability for driver in after.drivers) == pytest.approx(1.0)

    database.dispose()
