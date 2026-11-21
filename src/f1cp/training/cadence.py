"""When retraining is worth doing at all.

Retraining after every Grand Prix looks like the obvious way to keep a model current, and
it is useless here. :func:`~f1cp.training.splitters.completed_seasons` admits only seasons
whose whole calendar has been run: until the final round, every row of the season in
progress reads ``champion = 0`` because nobody has won yet, so that season is an inference
target and never a training one. A mid-season retrain therefore fits exactly the data the
previous run fitted, and appends six identical candidates to the audit history for nothing.

New rounds still reach the served probabilities without a retrain, because inference
rebuilds features from the live database (see :mod:`f1cp.inference.predictor`).

The one moment that adds training signal is a season closing: it is the only event that
creates a new label. So retraining is *due* exactly when the most recent fully-run season
ended after the last training run that succeeded. Comparing those two dates, rather than
remembering which seasons a run saw, needs nothing beyond what is already stored and stays
correct if a run is started by hand in between.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1cp.domain.enums import RunStatus

if TYPE_CHECKING:
    import datetime as dt

    from f1cp.db.models import Race
    from f1cp.db.uow import UnitOfWork

__all__ = ["is_due", "last_closed_season_final", "retraining_due"]

# How many recent runs to scan for the last successful one. Failed runs are rare and every
# run is recorded, so a short window is plenty.
_RECENT_RUNS = 50


def last_closed_season_final(uow: UnitOfWork) -> Race | None:
    """The final race of the most recent season whose whole calendar has results.

    Args:
        uow: An open unit of work.

    Returns:
        That season's last race, or ``None`` if no stored season is complete.
    """
    for season in reversed(uow.races.seasons()):
        calendar = uow.races.for_season(season)
        if not calendar:
            continue
        played = set(uow.races.rounds_with_results(season))
        if played and all(race.round in played for race in calendar):
            return calendar[-1]
    return None


def is_due(*, season_closed_on: dt.date | None, last_trained_at: dt.datetime | None) -> bool:
    """Decide from the two dates alone, so the rule is testable without a database.

    Args:
        season_closed_on: Date of the final race of the latest complete season.
        last_trained_at: When the latest successful training run finished.

    Returns:
        Whether a closed season has not been trained on yet.
    """
    if season_closed_on is None:
        return False
    if last_trained_at is None:
        return True
    # On the day of the final race itself, results may well have arrived after the run.
    return last_trained_at.date() <= season_closed_on


def retraining_due(uow: UnitOfWork) -> bool:
    """Whether a season has closed since the last successful training run.

    Args:
        uow: An open unit of work.

    Returns:
        ``True`` if a retrain would add a season the current models have not seen.
    """
    final = last_closed_season_final(uow)
    last_success = next(
        (
            run
            for run in uow.training_runs.recent(limit=_RECENT_RUNS)
            if run.status == str(RunStatus.SUCCEEDED)
        ),
        None,
    )
    return is_due(
        season_closed_on=final.date if final is not None else None,
        last_trained_at=(
            (last_success.finished_at or last_success.started_at) if last_success else None
        ),
    )
