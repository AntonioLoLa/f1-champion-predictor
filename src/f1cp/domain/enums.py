"""Shared enumerations."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["IngestionMode", "ProviderName", "RunStatus", "SeasonPhase", "TableName"]


class ProviderName(StrEnum):
    """Identifiers of the supported data providers."""

    JOLPICA = "jolpica"
    F1DB = "f1db"
    FASTF1 = "fastf1"
    OPENF1 = "openf1"
    KAGGLE = "kaggle"


class IngestionMode(StrEnum):
    """How an ingestion run acquires data."""

    BOOTSTRAP = "bootstrap"
    """Bulk-load full history from a downloadable release."""

    INCREMENTAL = "incremental"
    """Fetch only rounds not already stored."""

    REFRESH = "refresh"
    """Re-fetch a specific season regardless of what is stored."""


class RunStatus(StrEnum):
    """Lifecycle status of a background run (ingestion or training)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SeasonPhase(StrEnum):
    """Bucket describing how far into a season a round falls.

    Championship prediction is trivially easy late in a season and genuinely hard early,
    so every metric is reported per phase rather than pooled.
    """

    EARLY = "early"
    MID = "mid"
    LATE = "late"


class TableName(StrEnum):
    """Canonical table names shared by providers, the database and the feature layer."""

    SEASONS = "seasons"
    CIRCUITS = "circuits"
    DRIVERS = "drivers"
    CONSTRUCTORS = "constructors"
    RACES = "races"
    RACE_RESULTS = "race_results"
    QUALIFYING_RESULTS = "qualifying_results"
    SPRINT_RESULTS = "sprint_results"
    DRIVER_STANDINGS = "driver_standings"
    CONSTRUCTOR_STANDINGS = "constructor_standings"
    PIT_STOPS = "pit_stops"
