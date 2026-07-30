"""FastAPI application.

Builds the same composition root the CLI uses (:class:`~f1cp.container.AppContext`) once, at
startup, and stores it on ``app.state`` -- every router reaches its dependencies through
that (see :mod:`f1cp.api.deps`), never through module-level globals, so the whole
application graph can be rebuilt against a throwaway database in tests exactly the way the
CLI's is.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from f1cp.api.routers import (
    archive,
    contention,
    data,
    explain,
    health,
    model,
    predict,
    standings,
    train,
)
from f1cp.config import load_settings
from f1cp.container import AppContext

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from f1cp.config.settings import Settings

__all__ = ["app", "create_app"]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Pre-built settings, so tests can point the app at a throwaway database
            without relying on environment variables. Loaded from ``conf/`` when omitted.

    Returns:
        A configured application. The object graph itself (database, MLflow, the
        prediction service) is built lazily, in the lifespan below, so importing this
        module never touches a database or the filesystem.
    """
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        context = AppContext.create(settings=resolved, ensure_schema=True)
        app.state.context = context
        app.state.settings = context.settings
        app.state.database = context.database
        app.state.predictor = context.predictor()
        try:
            yield
        finally:
            context.close()

    app = FastAPI(
        title=resolved.api.title,
        version=resolved.api.version,
        lifespan=lifespan,
    )
    if resolved.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(train.router)
    app.include_router(data.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(standings.router)
    app.include_router(contention.router)
    app.include_router(archive.router)
    return app


app = create_app()
