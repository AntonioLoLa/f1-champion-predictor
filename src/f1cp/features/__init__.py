"""Feature engineering: the panel builder and the block pipeline.

Pure and network-free: every function here takes polars frames in and returns polars
frames out, with no dependency on :mod:`f1cp.db`. See :mod:`f1cp.features.panel` for the
leakage-safety guarantee the whole layer relies on.
"""

from f1cp.features.base import FeatureBlock
from f1cp.features.context import KEY_COLUMNS, FeatureContext, SourceFrames
from f1cp.features.pipeline import run_pipeline
from f1cp.features.registry import ALL_BLOCKS, enabled_blocks

__all__ = [
    "ALL_BLOCKS",
    "KEY_COLUMNS",
    "FeatureBlock",
    "FeatureContext",
    "SourceFrames",
    "enabled_blocks",
    "run_pipeline",
]
