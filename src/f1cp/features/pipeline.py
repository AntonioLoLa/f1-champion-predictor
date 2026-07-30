"""Feature pipeline execution.

Builds the panel, then runs every enabled block in order, checking each block's declared
dependencies before it runs. This is the single function the dataset builder and the
inference service both call, so the exact same feature logic that trained a model is what
serves its predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from f1cp.features.context import FeatureContext
from f1cp.features.panel import build_panel
from f1cp.features.registry import enabled_blocks
from f1cp.logging_config import get_logger

if TYPE_CHECKING:
    from f1cp.config.settings import FeaturesConfig, PathsConfig
    from f1cp.features.context import SourceFrames

__all__ = ["run_pipeline"]

logger = get_logger(__name__)


def run_pipeline(
    sources: SourceFrames,
    config: FeaturesConfig,
    paths: PathsConfig,
) -> pl.DataFrame:
    """Build the panel and apply every enabled feature block.

    Args:
        sources: Curated Formula 1 data.
        config: Feature engineering configuration.
        paths: Filesystem layout, passed through to blocks that read side files.

    Returns:
        The fully engineered feature matrix, one row per ``(season, round, driver_id)``.
    """
    panel = build_panel(sources)
    context = FeatureContext(panel=panel, sources=sources, config=config, paths=paths)

    frame = panel
    available: set[str] = set(frame.columns)
    for block in enabled_blocks(config):
        block.check_requirements(available)
        logger.info("Running feature block", extra={"block": block.name})
        frame = block.build(frame, context)
        available = set(frame.columns)

    return frame.sort(["season", "round", "driver_id"])
