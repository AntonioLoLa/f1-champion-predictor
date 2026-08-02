"""Dataset construction: source data in, a versioned Parquet feature matrix out.

Ties together the database loader, the feature pipeline and the schema contract, and is
the only place that decides what a "row" of the model-ready dataset actually is: a
``(season, round, driver_id)`` triple within the configured season range, restricted to
rounds the driver actually contested, labelled with whether they went on to win that
season's title.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from f1cp.dataset.loader import load_source_frames
from f1cp.dataset.schema import validate_dataset
from f1cp.features.context import KEY_COLUMNS
from f1cp.features.pipeline import run_pipeline
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.config.settings import Settings
    from f1cp.db.uow import UnitOfWork

__all__ = ["DatasetArtifact", "DatasetContractError", "build_dataset"]

logger = get_logger(__name__)


class DatasetContractError(ValueError):
    """Raised when a built dataset fails :func:`~f1cp.dataset.schema.validate_dataset`."""


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """A built dataset plus the metadata needed to register it.

    Attributes:
        frame: The engineered, filtered, labelled feature matrix.
        version: Content-addressed version string (``<config_hash>-<build_date>``).
        path: Where the frame was written as Parquet.
        feature_names: Columns that are model inputs (excludes keys, metadata and target).
        min_season: Earliest season present.
        max_season: Latest season present.
        config_hash: Hash of the feature configuration that produced this dataset.
    """

    frame: pl.DataFrame
    version: str
    path: str
    feature_names: list[str]
    min_season: int
    max_season: int
    config_hash: str

    @property
    def n_rows(self) -> int:
        """Number of rows in the dataset."""
        return self.frame.height

    @property
    def n_features(self) -> int:
        """Number of model-input columns."""
        return len(self.feature_names)


def build_dataset(uow: UnitOfWork, settings: Settings) -> DatasetArtifact:
    """Run the full dataset build: load, engineer, filter, label, validate, persist.

    Args:
        uow: An open :class:`~f1cp.db.uow.UnitOfWork` to read curated data from.
        settings: Application settings.

    Returns:
        The built :class:`DatasetArtifact`.

    Raises:
        DatasetContractError: If the built frame fails the schema contract.
    """
    sources = load_source_frames(uow, max_season=settings.data.resolved_max_season)
    engineered = run_pipeline(sources, settings.features, settings.paths)

    labelled = _attach_target(engineered, uow, settings.features.target_column)
    filtered = _apply_filters(labelled, settings)

    metadata_columns = set(settings.features.metadata_columns) | set(KEY_COLUMNS)
    feature_names = sorted(
        column
        for column in filtered.columns
        if column not in metadata_columns and column != settings.features.target_column
    )

    violations = validate_dataset(
        filtered,
        target_column=settings.features.target_column,
        metadata_columns=settings.features.metadata_columns,
    )
    if violations:
        detail = "\n".join(f"  - {violation}" for violation in violations)
        msg = f"Dataset failed schema contract:\n{detail}"
        raise DatasetContractError(msg)

    config_hash = _hash_config(settings)
    min_season = int(filtered["season"].min())  # type: ignore[arg-type]
    max_season = int(filtered["season"].max())  # type: ignore[arg-type]
    version = f"{config_hash[:12]}-{dt.date.today().isoformat()}"

    settings.paths.ensure_exist()
    path = settings.paths.processed / f"dataset_{version}.parquet"
    filtered.write_parquet(path)
    logger.info(
        "Built dataset",
        extra={
            "version": version,
            "rows": filtered.height,
            "features": len(feature_names),
            "seasons": f"{min_season}-{max_season}",
        },
    )

    return DatasetArtifact(
        frame=filtered,
        version=version,
        path=str(path),
        feature_names=feature_names,
        min_season=min_season,
        max_season=max_season,
        config_hash=config_hash,
    )


def _attach_target(frame: pl.DataFrame, uow: UnitOfWork, target_column: str) -> pl.DataFrame:
    """Label every row with whether its driver went on to win that season's title."""
    champions = uow.standings.champion_by_season()
    if not champions:
        return frame.with_columns(pl.lit(0).alias(target_column))

    champions_frame = pl.DataFrame(
        {"season": list(champions.keys()), "_champion_driver_id": list(champions.values())},
        schema={"season": pl.Int64, "_champion_driver_id": pl.String},
    )
    return (
        frame.join(champions_frame, on="season", how="left")
        .with_columns(
            (pl.col("driver_id") == pl.col("_champion_driver_id"))
            .fill_null(value=False)
            .cast(pl.Int8)
            .alias(target_column)
        )
        .drop("_champion_driver_id")
    )


def _apply_filters(frame: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Apply the season-range bound and the row-level filters from configuration.

    Deliberately *not* filtered here: rows where ``entered`` is false, i.e. rounds a driver
    sat out while still contesting the championship. Dropping them was the single most
    damaging defect this project has had, and it is worth recording why.

    The question the model answers is "after round R, who wins the title", and the answer
    set is *everyone in the championship*, not everyone who started that Grand Prix.
    Missing a race does not remove a driver from the standings. Dropping those rows meant
    training and evaluation scored a population that inference never sees -- and, worse,
    that :class:`~f1cp.models.normalizer.GroupProbabilityNormalizer` could be handed a
    group with no champion in it at all. At the 2020 Sakhir Grand Prix (round 16 of 17)
    Hamilton missed the race with COVID while leading the championship by 127 points. His
    row was dropped, every remaining driver's calibrated probability was correctly near
    zero -- and rescaling those to sum to one reported a 76% title probability for a
    mathematically eliminated Bottas, which no metric in the project could see, because
    :func:`~f1cp.training.metrics.group_log_loss` skips groups with no positive row.

    Keeping the rows costs nothing elsewhere: the panel is dense *before* this filter, so
    every rolling window already counted the missed round as a zero. Only membership of the
    training set and of the normalisation group changes.
    """
    filters = settings.features.filters
    out = frame.filter(
        (pl.col("season") >= settings.data.min_season)
        & (pl.col("season") <= settings.data.resolved_max_season)
    )

    if "rounds_total" in out.columns:
        out = out.filter(pl.col("rounds_total") >= filters.min_rounds_in_season)

    return out


def _hash_config(settings: Settings) -> str:
    """Hash the parts of configuration that determine dataset content.

    Args:
        settings: Application settings.

    Returns:
        A hex-encoded SHA-256 digest, stable across runs given identical configuration.
    """
    payload = {
        "features": settings.features.model_dump(mode="json"),
        "min_season": settings.data.min_season,
        "max_season": settings.data.resolved_max_season,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
