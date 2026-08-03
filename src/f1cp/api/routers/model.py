"""Model version listing and activation endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from f1cp.api.deps import get_database, require_write_access
from f1cp.api.schemas.model import ActivateModelResponse, ModelVersionSummary

if TYPE_CHECKING:
    from f1cp.db.models import ModelVersion
    from f1cp.db.session import Database

__all__ = ["router"]

router = APIRouter(prefix="/model", tags=["model"])


@router.get("", response_model=ModelVersionSummary)
def active_model(database: Database = Depends(get_database)) -> ModelVersionSummary:
    """Return the model version currently serving predictions.

    Args:
        database: The shared database handle.

    Returns:
        The active model version.

    Raises:
        HTTPException: 404 if no model has ever been activated.
    """
    with database.session_scope() as uow:
        active = uow.models.get_active()
        if active is None:
            msg = "No active model; run `f1cp train run` first"
            raise HTTPException(status_code=404, detail=msg)
        return _to_summary(active)


@router.get("/versions", response_model=list[ModelVersionSummary])
def list_versions(database: Database = Depends(get_database)) -> list[ModelVersionSummary]:
    """List every trained model version, newest first.

    Args:
        database: The shared database handle.

    Returns:
        Every stored model version.
    """
    with database.session_scope() as uow:
        return [_to_summary(version) for version in uow.models.list_versions()]


@router.post(
    "/{model_version_id}/activate",
    response_model=ActivateModelResponse,
    dependencies=[Depends(require_write_access)],
)
def activate(
    model_version_id: int, database: Database = Depends(get_database)
) -> ActivateModelResponse:
    """Make one model version the one serving predictions.

    Args:
        model_version_id: Primary key of the version to activate.
        database: The shared database handle.

    Returns:
        The now-active model version.

    Raises:
        HTTPException: 404 if no such model version exists, 409 if it has no logged model
            to serve, 403 if the service is configured read-only.
    """
    with database.session_scope() as uow:
        candidate = uow.models.get(model_version_id)
        if candidate is None:
            msg = f"Model version {model_version_id} does not exist"
            raise HTTPException(status_code=404, detail=msg)
        if candidate.mlflow_model_uri is None:
            msg = (
                f"Model version {model_version_id} has no logged model and cannot serve predictions"
            )
            raise HTTPException(status_code=409, detail=msg)
        try:
            version = uow.models.activate(model_version_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ActivateModelResponse(id=version.id, name=version.name, is_active=version.is_active)


def _to_summary(version: ModelVersion) -> ModelVersionSummary:
    """Project a :class:`~f1cp.db.models.ModelVersion` onto its API summary."""
    return ModelVersionSummary(
        id=version.id,
        training_run_id=version.training_run_id,
        name=version.name,
        algorithm=version.algorithm,
        is_active=version.is_active,
        is_best=version.is_best,
        metrics=version.metrics,
        created_at=version.created_at,
    )
