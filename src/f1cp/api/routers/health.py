"""Health check endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from f1cp.api.deps import get_database, get_settings
from f1cp.api.schemas.health import HealthResponse

if TYPE_CHECKING:
    from f1cp.config.settings import Settings
    from f1cp.db.session import Database

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Report database connectivity and whether a model is active to serve.

    Args:
        database: The shared database handle.
        settings: Effective application settings.

    Returns:
        A liveness/readiness summary. Never raises: a database failure is reported as
        ``status="degraded"`` rather than a 500, since this endpoint exists precisely to be
        checked when something else might be down.
    """
    try:
        with database.session_scope() as uow:
            active = uow.models.get_active()
        database_ok = True
    except Exception:  # health checks must never raise
        database_ok = False
        active = None

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database=database_ok,
        active_model=active.name if active else None,
        read_only=settings.api.read_only,
    )
