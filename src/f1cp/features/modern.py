"""Modern-era enrichment features from FastF1 and OpenF1.

Off by default (see ``conf/features.yaml``): FastF1 telemetry only covers 2018 onwards and
OpenF1 only 2023 onwards, and a first run should never block on a telemetry download. When
enabled, this block reads the enrichment Parquet files written by ``f1cp data enrich``
(see :meth:`f1cp.data.ingest.DataIngestor.enrich`) rather than calling the providers
directly, keeping the feature pipeline itself network-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.base import FeatureBlock
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.features.context import FeatureContext

__all__ = ["ModernBlock"]

logger = get_logger(__name__)

_PLACEHOLDER_COLUMNS: tuple[tuple[str, type[pl.DataType]], ...] = (
    ("race_pace_delta_s", pl.Float64),
    ("avg_pit_stop_duration_s", pl.Float64),
    ("pit_stop_count", pl.Int64),
)


class ModernBlock(FeatureBlock):
    """Race-pace, stint and pit-stop features for 2018+ seasons."""

    name = "modern"
    requires = ("season", "round", "driver_id")

    def build(self, frame: pl.DataFrame, context: FeatureContext) -> pl.DataFrame:
        """Left-join pre-computed enrichment metrics onto rows within the covered eras.

        Args:
            frame: The panel built so far.
            context: Shared inputs; reads ``paths.interim`` for the enrichment files
                written by ``f1cp data enrich fastf1`` / ``f1cp data enrich openf1``.

        Returns:
            ``frame`` with enrichment columns appended (null outside the covered seasons
            or when no enrichment file has been generated yet).
        """
        block_config = context.config.blocks.modern
        enrichment = _load_enrichment(context)
        if enrichment is None or not enrichment.height:
            logger.info("No enrichment data available; modern block columns will be entirely null")
            return frame.with_columns(
                pl.lit(None, dtype=dtype).alias(name) for name, dtype in _PLACEHOLDER_COLUMNS
            )

        joined = frame.join(enrichment, on=["season", "round", "driver_id"], how="left")
        value_columns = [c for c in enrichment.columns if c not in ("season", "round", "driver_id")]
        return joined.with_columns(
            pl.when(pl.col("season") >= block_config.min_season).then(pl.col(column)).alias(column)
            for column in value_columns
        )


def _load_enrichment(context: FeatureContext) -> pl.DataFrame | None:
    """Load and concatenate whichever enrichment Parquet files are on disk.

    Args:
        context: Shared inputs; ``paths.interim`` is where ``f1cp data enrich`` writes.

    Returns:
        A frame keyed by ``(season, round, driver_id)``, or ``None`` if nothing has been
        generated yet.
    """
    frames: list[pl.DataFrame] = []
    for provider_name in ("fastf1", "openf1"):
        path = context.paths.interim / f"enrichment_{provider_name}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")
