"""Feature block discovery and ordering.

Order matters: ``standings`` produces ``current_points`` and ``gap_to_leader``, which
``season_context`` and ``constructor`` both read; ``form``/``rates`` read ``is_win`` and
friends from the panel itself. Declaring the order once here, rather than inferring it from
``requires``, keeps the pipeline predictable and the failure mode (a missing dependency)
loud rather than a silent topological-sort surprise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1cp.features.base import FeatureBlock
from f1cp.features.circuit import CircuitBlock
from f1cp.features.constructor import ConstructorBlock
from f1cp.features.driver import DriverBlock
from f1cp.features.form import FormBlock
from f1cp.features.modern import ModernBlock
from f1cp.features.qualifying import QualifyingBlock
from f1cp.features.rates import RatesBlock
from f1cp.features.season_context import SeasonContextBlock
from f1cp.features.standings import StandingsBlock

if TYPE_CHECKING:
    from f1cp.config.settings import FeaturesConfig

__all__ = ["ALL_BLOCKS", "enabled_blocks"]

# Declaration order is execution order.
ALL_BLOCKS: tuple[FeatureBlock, ...] = (
    StandingsBlock(),
    SeasonContextBlock(),
    FormBlock(),
    RatesBlock(),
    QualifyingBlock(),
    ConstructorBlock(),
    DriverBlock(),
    CircuitBlock(),
    ModernBlock(),
)


def enabled_blocks(config: FeaturesConfig) -> list[FeatureBlock]:
    """Resolve the blocks switched on in configuration, in execution order.

    Args:
        config: Feature engineering configuration.

    Returns:
        The enabled blocks, in the fixed order declared by :data:`ALL_BLOCKS`.
    """
    enabled_names = set(config.blocks.enabled_names())
    return [block for block in ALL_BLOCKS if block.name in enabled_names]
