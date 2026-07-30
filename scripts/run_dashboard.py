#!/usr/bin/env python
"""Run the Streamlit dashboard.

    python scripts/run_dashboard.py [--host HOST] [--port PORT]

Talks to the FastAPI service configured by ``dashboard.api_base_url`` (see
``conf/config.yaml`` / ``API_BASE_URL``); the API must already be running. A thin wrapper
over ``streamlit run``, so there is exactly one entrypoint for the dashboard process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from f1cp.config import load_settings

_APP_PATH = Path(__file__).resolve().parent.parent / "src" / "f1cp" / "dashboard" / "app.py"


def main() -> None:
    """Parse arguments and launch ``streamlit run`` as a subprocess."""
    settings = load_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.dashboard.host, help="Bind address.")
    parser.add_argument("--port", type=int, default=settings.dashboard.port, help="Bind port.")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(_APP_PATH),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
