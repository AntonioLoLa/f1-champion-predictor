"""Entrypoint for Streamlit Community Cloud's free tier.

Community Cloud runs exactly one Python process and injects no other services, unlike
``docker-compose.yml``'s four. The dashboard still talks to the API over HTTP only (see
``f1cp.dashboard.app``) -- that boundary is unchanged -- but something has to bring the API
up first, in the same process, since there is nowhere else for it to run. That is this
file's only job; it is deployment glue, not application code, which is why it lives at the
repo root rather than inside ``src/f1cp``.

Two things only this entrypoint needs to do, that the Docker deployment does not:

- Point every path-based setting at *this* checkout's ``data/`` directory with an absolute
  path, since Community Cloud's working directory at process start is not guaranteed to be
  the repo root.
- Rewrite the absolute artifact paths MLflow baked into ``data/mlflow.db`` at training time,
  on the developer's machine, to wherever this checkout actually landed. Without that,
  ``mlflow.sklearn.load_model`` looks for model files on a filesystem that doesn't exist
  here. There is no live MLflow server to reissue those paths from -- the sqlite file
  bundled into the repo is the only copy -- so it is patched in place, once, before anything
  reads it.
"""

from __future__ import annotations

import os
import runpy
import socket
import sqlite3
import threading
import time
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"

# The absolute prefix every artifact path in data/mlflow.db was recorded with, back when
# the model zoo was trained on the development machine. Constant, not derived: rewriting
# depends on knowing what to replace, and that never changes short of retraining.
_TRAINING_MACHINE_PREFIX = "/home/antonio/projects/f1/champion-predictor"

# conf/config.yaml's defaults are relative (`sqlite:///data/f1cp.db`, `sqlite:///./data/
# mlflow.db`), which is exactly what Docker Compose overrides via these same variables --
# so setting them here follows an existing pattern rather than inventing one.
os.environ.setdefault("F1CP_DATA_DIR", str(_DATA_DIR))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DATA_DIR / 'f1cp.db'}")
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_DATA_DIR / 'mlflow.db'}")
os.environ.setdefault("API_BASE_URL", "http://127.0.0.1:8000")

_API_HOST = "127.0.0.1"
_API_PORT = 8000

# (table, column) pairs in data/mlflow.db known to hold an absolute artifact path.
_ARTIFACT_PATH_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
)


def _rewrite_artifact_paths() -> None:
    """Repoint data/mlflow.db's stored artifact paths at this checkout's location."""
    con = sqlite3.connect(_DATA_DIR / "mlflow.db")
    try:
        new_prefix = str(_REPO_ROOT)
        for table, column in _ARTIFACT_PATH_COLUMNS:
            con.execute(
                f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?) "  # noqa: S608
                f"WHERE {column} LIKE ?",
                (_TRAINING_MACHINE_PREFIX, new_prefix, f"{_TRAINING_MACHINE_PREFIX}%"),
            )
        con.commit()
    finally:
        con.close()


def _start_api() -> None:
    """Run the FastAPI app in a background thread and block until it accepts connections."""

    def _run() -> None:
        import uvicorn

        uvicorn.run("f1cp.api.main:app", host=_API_HOST, port=_API_PORT, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex((_API_HOST, _API_PORT)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError("Embedded API did not come up within 30s.")


@st.cache_resource
def _bootstrap() -> None:
    """Run the one-time setup exactly once per process, however many reruns follow."""
    _rewrite_artifact_paths()
    _start_api()


_bootstrap()

# Streamlit reruns *this* file from scratch on every widget interaction, but a plain
# `import f1cp.dashboard.app` would only execute its module body -- and therefore its
# `st.navigation(...).run()` -- once, on the very first run, since Python caches imported
# modules. `runpy` re-executes the file fresh every time, exactly as `streamlit run` would
# if pointed at it directly.
runpy.run_path(str(_REPO_ROOT / "src" / "f1cp" / "dashboard" / "app.py"), run_name="__main__")
