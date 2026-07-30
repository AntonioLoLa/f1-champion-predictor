"""Logging setup.

Two formats are supported: a human-readable console format for local development and a
single-line JSON format for containerised deployments where logs are shipped to a
collector.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["JsonFormatter", "configure_logging", "get_logger"]

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Attributes present on every LogRecord; anything else was supplied via ``extra=``.
_RESERVED_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string.

        Args:
            record: The record to format.

        Returns:
            A single-line JSON document.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_ATTRS
        }
        if extras:
            payload["context"] = extras

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure the root logger.

    Safe to call more than once: existing handlers are replaced rather than duplicated,
    which matters because Streamlit and Uvicorn both re-import application modules.

    Args:
        level: Logging level name, e.g. ``"INFO"``.
        fmt: Either ``"console"`` or ``"json"``.
    """
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # These libraries are extremely chatty at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "matplotlib", "fastf1", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
