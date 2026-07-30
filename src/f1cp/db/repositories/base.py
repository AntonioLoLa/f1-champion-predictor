"""Repository foundations.

Provides a typed generic repository plus a dialect-aware bulk upsert. Ingestion is
inherently re-runnable -- the same round may be fetched again after a correction to the
official results -- so every write goes through an idempotent upsert rather than a plain
insert.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from f1cp.db.models import Base
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

__all__ = ["AbstractRepository", "SqlAlchemyRepository", "bulk_upsert"]

logger = get_logger(__name__)

# Chunk size for multi-row statements. Comfortably under SQLite's bound-parameter ceiling
# even for the widest table, while keeping round-trips to PostgreSQL low.
_CHUNK_ROWS = 500


class AbstractRepository[ModelT: Base](ABC):
    """Persistence interface for one aggregate.

    Callers depend on this abstraction rather than on SQLAlchemy, which is what lets the
    API be tested against fakes and keeps the storage engine swappable.
    """

    @abstractmethod
    def add(self, entity: ModelT) -> ModelT:
        """Persist a new entity."""

    @abstractmethod
    def get(self, entity_id: Any) -> ModelT | None:
        """Fetch an entity by primary key, or ``None`` if absent."""

    @abstractmethod
    def list(self, *, limit: int | None = None) -> Sequence[ModelT]:
        """Return stored entities."""


class SqlAlchemyRepository[ModelT: Base](AbstractRepository[ModelT]):
    """SQLAlchemy-backed implementation of :class:`AbstractRepository`."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, entity: ModelT) -> ModelT:
        """Persist a new entity and flush so its generated key is populated.

        Args:
            entity: The ORM instance to persist.

        Returns:
            The same instance, now attached to the session.
        """
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, entity_id: Any) -> ModelT | None:
        """Fetch an entity by primary key.

        Args:
            entity_id: Primary key value.

        Returns:
            The entity, or ``None`` if it does not exist.
        """
        return self.session.get(self.model, entity_id)

    def list(self, *, limit: int | None = None) -> Sequence[ModelT]:
        """Return stored entities.

        Args:
            limit: Optional maximum number of rows.

        Returns:
            A sequence of entities.
        """
        statement = select(self.model)
        if limit is not None:
            statement = statement.limit(limit)
        return self.session.execute(statement).scalars().all()

    def count(self) -> int:
        """Return the number of stored rows."""
        from sqlalchemy import func

        return self.session.execute(select(func.count()).select_from(self.model)).scalar_one()


def bulk_upsert(
    session: Session,
    model: type[Base],
    rows: Iterable[dict[str, Any]],
    *,
    index_elements: Sequence[str],
    update_columns: Sequence[str] | None = None,
) -> int:
    """Insert rows, updating any that collide on ``index_elements``.

    Both PostgreSQL and SQLite support ``ON CONFLICT DO UPDATE``, so a single code path
    covers the Compose stack and a local file database.

    Args:
        session: Active session.
        model: ORM model to write to.
        rows: Row dictionaries keyed by column name.
        index_elements: Columns forming the conflict target (a primary key or unique
            constraint).
        update_columns: Columns to overwrite on conflict. Defaults to every supplied
            column that is not part of the conflict target.

    Returns:
        The number of rows submitted.

    Raises:
        NotImplementedError: If the session is bound to an unsupported dialect.
    """
    materialised = list(rows)
    if not materialised:
        return 0

    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        insert = pg_insert
    elif dialect == "sqlite":
        insert = sqlite_insert  # type: ignore[assignment]
    else:  # pragma: no cover - guarded by configuration
        msg = f"bulk_upsert does not support the {dialect!r} dialect"
        raise NotImplementedError(msg)

    # PostgreSQL rejects a statement that would update the same row twice, so rows
    # colliding on the conflict target must be collapsed first. This is not hypothetical:
    # shared drives in the 1950s put the same driver in one race twice.
    deduplicated = _deduplicate(materialised, index_elements)

    conflict_target = set(index_elements)
    columns = update_columns or [
        column for column in deduplicated[0] if column not in conflict_target
    ]

    for start in range(0, len(deduplicated), _CHUNK_ROWS):
        chunk = deduplicated[start : start + _CHUNK_ROWS]
        statement = insert(model).values(chunk)
        if columns:
            statement = statement.on_conflict_do_update(
                index_elements=list(index_elements),
                set_={name: getattr(statement.excluded, name) for name in columns},
            )
        else:
            statement = statement.on_conflict_do_nothing(index_elements=list(index_elements))
        session.execute(statement)

    return len(deduplicated)


def _deduplicate(rows: list[dict[str, Any]], index_elements: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse rows sharing the same conflict-target values, keeping the last.

    Args:
        rows: Candidate rows.
        index_elements: Columns forming the conflict target.

    Returns:
        Rows with unique conflict-target values, in first-seen order.
    """
    seen: dict[tuple[Any, ...], int] = {}
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(column) for column in index_elements)
        if (position := seen.get(key)) is not None:
            unique[position] = row
        else:
            seen[key] = len(unique)
            unique.append(row)

    if (dropped := len(rows) - len(unique)) > 0:
        logger.debug("Collapsed duplicate rows before upsert", extra={"dropped": dropped})
    return unique
