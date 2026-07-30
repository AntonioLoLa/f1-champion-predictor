"""Alembic environment.

The database URL always comes from application settings rather than ``alembic.ini``, so
migrations target the same database as the rest of the system in every environment.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from f1cp.config import get_settings
from f1cp.db.models import Base

config = context.config
target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database.url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=settings.database.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    section = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most things in place; batch mode rewrites the table.
            render_as_batch=settings.database.is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
