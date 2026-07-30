#!/usr/bin/env python
"""Run the FastAPI service.

    python scripts/serve_api.py [--reload] [--host HOST] [--port PORT]

Host and port default to ``conf/config.yaml``'s ``api.host``/``api.port`` (in turn
overridable via ``API_HOST``/``API_PORT``), so this and ``docker-compose.yml`` agree
without duplicating the values.
"""

from __future__ import annotations

import argparse

import uvicorn

from f1cp.config import load_settings


def main() -> None:
    """Parse arguments and run the Uvicorn server."""
    settings = load_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.api.host, help="Bind address.")
    parser.add_argument("--port", type=int, default=settings.api.port, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    uvicorn.run(
        "f1cp.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
