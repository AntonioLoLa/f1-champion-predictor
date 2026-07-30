"""Season-phase bucketing.

A model that only ever gets judged on pooled metrics looks far better than it is: by the
final rounds the championship is often mathematically decided, so those rows are trivially
classifiable. Splitting metrics by how far the season had progressed is what makes the
evaluation honest.
"""

from __future__ import annotations

from collections.abc import Mapping

from f1cp.domain.enums import SeasonPhase

__all__ = ["DEFAULT_PHASE_BINS", "phase_for_progress"]

DEFAULT_PHASE_BINS: Mapping[str, tuple[float, float]] = {
    SeasonPhase.EARLY: (0.0, 0.34),
    SeasonPhase.MID: (0.34, 0.67),
    SeasonPhase.LATE: (0.67, 1.01),
}


def phase_for_progress(
    progress: float,
    bins: Mapping[str, tuple[float, float]] | None = None,
) -> SeasonPhase:
    """Map a season-progress fraction onto a :class:`SeasonPhase`.

    Args:
        progress: Fraction of the season completed, ``round / total_rounds`` in ``(0, 1]``.
        bins: Optional custom bin edges, as ``{phase: (low, high)}`` with a half-open
            ``[low, high)`` convention. Defaults to :data:`DEFAULT_PHASE_BINS`.

    Returns:
        The phase whose bin contains ``progress``. Values at or above the last upper edge
        are clamped to :attr:`SeasonPhase.LATE`.
    """
    edges = bins if bins is not None else DEFAULT_PHASE_BINS
    for name, (low, high) in edges.items():
        if low <= progress < high:
            return SeasonPhase(name)
    return SeasonPhase.LATE
