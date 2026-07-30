"""Training endpoints.

Training a full model zoo takes minutes, so ``POST /train`` only schedules the run: the
run's external id is generated up front and handed back immediately, before the background
task starts, so a client can poll ``GET /train/{run_uuid}`` without a race against the row
actually being written.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from f1cp.api.deps import get_context, get_database
from f1cp.api.schemas.train import TrainRequest, TrainResponse, TrainStatusResponse
from f1cp.domain.enums import RunStatus
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.container import AppContext
    from f1cp.db.session import Database

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/train", tags=["train"])


def _run_training(context: AppContext, dataset_version: str | None, run_uuid: str) -> None:
    """Run training in the background. Failures are already recorded on the row itself."""
    try:
        context.trainer().run(dataset_version=dataset_version, run_uuid=run_uuid)
    except Exception:
        logger.exception("Background training run failed", extra={"run_uuid": run_uuid})


@router.post("", response_model=TrainResponse, status_code=202)
def train(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_context),
) -> TrainResponse:
    """Schedule a training run in the background.

    Args:
        payload: Training options.
        background_tasks: FastAPI's background task queue.
        context: The application's composition root.

    Returns:
        The scheduled run's external id, for polling via ``GET /train/{run_uuid}``.
    """
    run_uuid = str(uuid.uuid4())
    background_tasks.add_task(_run_training, context, payload.dataset_version, run_uuid)
    return TrainResponse(run_uuid=run_uuid, status=str(RunStatus.PENDING))


@router.get("/{run_uuid}", response_model=TrainStatusResponse)
def train_status(run_uuid: str, database: Database = Depends(get_database)) -> TrainStatusResponse:
    """Poll a scheduled training run.

    Args:
        run_uuid: External id returned by ``POST /train``.
        database: The shared database handle.

    Returns:
        The run's current status.

    Raises:
        HTTPException: 404 if no run has been recorded under that id yet (it may not have
            started; try again shortly), or it truly does not exist.
    """
    with database.session_scope() as uow:
        run = uow.training_runs.get_by_uuid(run_uuid)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No training run '{run_uuid}'")

        best_model_version_id = next(
            (version.id for version in uow.models.for_training_run(run.id) if version.is_best),
            None,
        )
        return TrainStatusResponse(
            run_uuid=run.run_uuid,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error=run.error,
            best_model_version_id=best_model_version_id,
        )
