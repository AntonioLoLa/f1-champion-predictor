"""Retraining cadence against a real database seeded with a season still in progress."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest

from f1cp.db.models import TrainingRun
from f1cp.db.session import Database
from f1cp.domain.enums import RunStatus
from f1cp.training.cadence import last_closed_season_final, retraining_due
from tests.integration.test_dataset_builder import _seed

if TYPE_CHECKING:
    from f1cp.config.settings import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings.database)
    db.create_all()
    # 2021 is seeded with only its first two rounds run.
    _seed(db)
    return db


def _add_run(database: Database, *, status: RunStatus, finished: dt.datetime) -> None:
    with database.session_scope() as uow:
        uow.training_runs.add(
            TrainingRun(
                run_uuid=f"run-{status}-{finished.isoformat()}",
                status=str(status),
                started_at=finished,
                finished_at=finished,
            )
        )


def test_the_season_in_progress_is_skipped(database: Database) -> None:
    with database.session_scope() as uow:
        final = last_closed_season_final(uow)
        assert final is not None
        assert final.season == 2020
        assert final.round == max(race.round for race in uow.races.for_season(2020))


def test_due_until_a_successful_run_follows_the_closed_season(database: Database) -> None:
    with database.session_scope() as uow:
        assert retraining_due(uow)

    after_close = dt.datetime(2021, 1, 15, tzinfo=dt.UTC)
    _add_run(database, status=RunStatus.FAILED, finished=after_close)
    with database.session_scope() as uow:
        assert retraining_due(uow), "a failed run must not count as having retrained"

    _add_run(database, status=RunStatus.SUCCEEDED, finished=after_close)
    with database.session_scope() as uow:
        assert not retraining_due(uow)
