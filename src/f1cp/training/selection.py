"""Model selection: choosing one winner from the zoo's trained candidates.

Accuracy would pick "always predict not champion" (see :mod:`f1cp.training.metrics`), so
selection is driven entirely by configuration: a primary metric and direction from
``training.selection``, tie-broken in the configured order. Both configured tie-breakers
(Brier score, expected calibration error) are error metrics, so ties are always broken
lower-is-better regardless of the primary metric's direction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from f1cp.config.settings import SelectionConfig

__all__ = ["Scored", "select_best"]


class Scored(Protocol):
    """Anything with a name and a bundle of aggregated metrics can be selected."""

    @property
    def name(self) -> str:
        """The candidate's identifier, e.g. a model zoo entry name."""
        ...

    @property
    def metrics(self) -> dict[str, float | None]:
        """Aggregated metric values, keyed by metric name."""
        ...


def select_best[T: Scored](candidates: Sequence[T], config: SelectionConfig) -> T:
    """Pick the winning candidate by primary metric, tie-broken in configured order.

    Args:
        candidates: Every trained candidate, each with its aggregated CV metrics.
        config: ``training.selection`` settings: which metric decides, which direction is
            better, and which metrics break ties.

    Returns:
        The winning candidate.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        msg = "Cannot select a best candidate from an empty list"
        raise ValueError(msg)

    sign = 1.0 if config.direction == "minimize" else -1.0

    def sort_key(candidate: T) -> tuple[float, ...]:
        primary = candidate.metrics.get(config.primary_metric)
        key = [math.inf if primary is None else sign * primary]
        for tie_breaker in config.tie_breakers:
            value = candidate.metrics.get(tie_breaker)
            key.append(math.inf if value is None else value)
        return tuple(key)

    return min(candidates, key=sort_key)
