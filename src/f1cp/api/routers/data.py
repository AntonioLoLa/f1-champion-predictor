"""Data acquisition endpoint. Mirrors ``f1cp data update``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, Depends

from f1cp.api.deps import get_context
from f1cp.api.schemas.data import IngestionReportResponse, UpdateDataRequest, UpdateDataResponse
from f1cp.domain.enums import RunStatus
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.container import AppContext
    from f1cp.data.ingest import IngestionReport

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["data"])


def _execute(context: AppContext, payload: UpdateDataRequest) -> list[IngestionReport]:
    """Run the requested bootstrap/refresh/incremental combination, in that order."""
    ingestor = context.ingestor()
    reports: list[IngestionReport] = []

    if payload.bootstrap:
        reports.append(ingestor.bootstrap(min_season=payload.min_season))

    if payload.refresh:
        for season in payload.seasons or []:
            reports.append(ingestor.refresh_season(season))
    elif not payload.bootstrap or payload.seasons:
        reports.append(ingestor.incremental(seasons=payload.seasons))

    return reports


def _run_in_background(context: AppContext, payload: UpdateDataRequest) -> None:
    """Run a data update in the background; failures are already on the audit rows."""
    try:
        _execute(context, payload)
    except Exception:
        logger.exception("Background data update failed")


def _to_response(report: IngestionReport) -> IngestionReportResponse:
    """Project an :class:`~f1cp.data.ingest.IngestionReport` onto its API response."""
    return IngestionReportResponse(
        provider=report.provider,
        mode=str(report.mode),
        seasons=report.seasons,
        rows_written=report.rows_written,
        status=str(report.status),
        error=report.error,
    )


@router.post("/update-data", response_model=UpdateDataResponse, status_code=202)
def update_data(
    payload: UpdateDataRequest,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_context),
) -> UpdateDataResponse:
    """Bootstrap and/or fetch new rounds of Formula 1 data.

    A bootstrap can take a long time (the full history), so ``background=True`` schedules
    it and returns immediately; the audit trail (`f1cp.db.models.DataIngestion`) is written
    by the ingestor itself as each provider call completes, and can be inspected via
    ``f1cp info`` or the database directly. A plain incremental update is normally fast
    enough (roughly seven requests per new round) to run inline.

    Args:
        payload: Update options, mirroring ``f1cp data update``.
        background_tasks: FastAPI's background task queue.
        context: The application's composition root.

    Returns:
        The completed reports, or an empty, ``"started"`` response when backgrounded.
    """
    if payload.background:
        background_tasks.add_task(_run_in_background, context, payload)
        return UpdateDataResponse(status="started", reports=[])

    reports = _execute(context, payload)
    status = (
        "failed" if any(report.status is RunStatus.FAILED for report in reports) else "completed"
    )
    return UpdateDataResponse(status=status, reports=[_to_response(r) for r in reports])
