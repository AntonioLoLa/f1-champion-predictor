"""FastAPI dependency providers.

Every dependency reaches into ``request.app.state``, populated once at startup by
:mod:`f1cp.api.main`'s lifespan. Routers never import a global engine or context, which is
what lets tests build a fresh application against a throwaway database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from f1cp.config.settings import Settings
    from f1cp.container import AppContext
    from f1cp.db.session import Database
    from f1cp.inference.predictor import PredictionService

__all__ = ["get_context", "get_database", "get_predictor", "get_settings"]


def get_context(request: Request) -> AppContext:
    """The application's composition root, built once at startup."""
    return request.app.state.context  # type: ignore[no-any-return]


def get_settings(request: Request) -> Settings:
    """Effective application settings."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_database(request: Request) -> Database:
    """The shared database handle. Routes open their own unit of work from it."""
    return request.app.state.database  # type: ignore[no-any-return]


def get_predictor(request: Request) -> PredictionService:
    """The prediction service, shared across requests so a served model loads once."""
    return request.app.state.predictor  # type: ignore[no-any-return]
