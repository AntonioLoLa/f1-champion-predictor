# syntax=docker/dockerfile:1
#
# Single multi-stage Dockerfile, one stage per deployable process (api, dashboard,
# pipeline), all built on a shared `base` stage. `docker-compose.yml` selects the target
# per service via `build.target`.
#
# A "shared base + 3 thin Dockerfiles" split was the original sketch, but plain Docker has
# no reliable way to make one Dockerfile's `FROM` reference an image another Dockerfile
# just built without either a manual pre-build step or version-sensitive BuildKit features
# -- `docker compose build` does not guarantee that ordering. A single file with named
# stages gets the same separation of concerns and actually works with a plain
# `docker compose up --build`.

FROM python:3.12-slim AS base

# libgomp1: XGBoost/LightGBM link against it at runtime (a classic "libgomp.so.1: cannot
# open shared object file" on slim Debian images without it). curl: container healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so an application-code change doesn't invalidate this layer.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md ./
COPY src/ src/
COPY conf/ conf/
COPY alembic.ini ./
# Everything the package needs is already in requirements.lock; --no-deps keeps this from
# re-resolving (and potentially drifting from) that lock.
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    F1CP_DATA_DIR=/app/data

# --------------------------------------------------------------------------------------
FROM base AS api

COPY scripts/serve_api.py scripts/
EXPOSE 8000
CMD ["python", "scripts/serve_api.py"]

# --------------------------------------------------------------------------------------
FROM base AS dashboard

COPY scripts/run_dashboard.py scripts/
EXPOSE 8501
CMD ["python", "scripts/run_dashboard.py"]

# --------------------------------------------------------------------------------------
FROM base AS pipeline

# No default entrypoint script: invoked as `docker compose run --rm pipeline <command>`
# for whichever pipeline stage is needed (update_data / build_dataset / train / evaluate).
COPY scripts/ scripts/
CMD ["f1cp", "info"]
