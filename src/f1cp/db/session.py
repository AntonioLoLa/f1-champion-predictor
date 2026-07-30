"""Engine, session and unit-of-work management.

A single :class:`Database` object owns the engine so that the API, the training job and the
tests can each build their own against different URLs without touching global state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from f1cp.db.models import Base
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.pool import ConnectionPoolEntry

    from f1cp.config.settings import DatabaseConfig
    from f1cp.db.uow import UnitOfWork

__all__ = ["Database"]

logger = get_logger(__name__)


class Database:
    """Owns the SQLAlchemy engine and hands out sessions."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine = self._create_engine(config)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False, autoflush=False
        )

    @staticmethod
    def _create_engine(config: DatabaseConfig) -> Engine:
        """Build an engine, adapting pool arguments to the backend."""
        kwargs: dict[str, Any] = {"echo": config.echo, "future": True}

        if config.is_sqlite:
            # SQLite's default pool takes none of the server-oriented pool arguments.
            engine = create_engine(config.url, **kwargs)

            @event.listens_for(engine, "connect")
            def _enable_foreign_keys(
                dbapi_connection: DBAPIConnection,
                _record: ConnectionPoolEntry,
            ) -> None:
                """SQLite ignores foreign keys unless they are switched on per connection."""
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            return engine

        kwargs |= {
            "pool_size": config.pool_size,
            "max_overflow": config.max_overflow,
            "pool_pre_ping": config.pool_pre_ping,
        }
        return create_engine(config.url, **kwargs)

    @property
    def engine(self) -> Engine:
        """The underlying SQLAlchemy engine."""
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Factory producing new sessions bound to this engine."""
        return self._session_factory

    def create_all(self) -> None:
        """Create any missing tables.

        Alembic owns schema migrations; this exists for tests and for the very first run
        against an empty database.
        """
        Base.metadata.create_all(self._engine)
        logger.debug("Ensured database schema exists", extra={"url": self._sanitised_url()})

    def drop_all(self) -> None:
        """Drop every table. Intended for tests."""
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session inside a transaction, committing on success.

        Yields:
            An open :class:`~sqlalchemy.orm.Session`.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def session_scope(self) -> Iterator[UnitOfWork]:
        """Yield a :class:`~f1cp.db.uow.UnitOfWork` bound to a fresh session.

        The common entry point for application code: one transactional scope exposing
        every repository.

        Yields:
            An open unit of work, committed on success and rolled back on error.
        """
        # Imported here because the unit of work depends on this module.
        from f1cp.db.uow import UnitOfWork

        with UnitOfWork(self) as uow:
            yield uow

    def _sanitised_url(self) -> str:
        """The connection URL with any password removed, safe to log."""
        url = self._config.url
        if "@" not in url or "://" not in url:
            return url
        scheme, rest = url.split("://", 1)
        credentials, host = rest.rsplit("@", 1)
        user = credentials.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"

    def dispose(self) -> None:
        """Close all pooled connections."""
        self._engine.dispose()
