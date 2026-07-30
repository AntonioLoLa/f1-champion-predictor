"""Domain layer: rules and vocabulary of the Formula 1 World Championship.

This layer has no dependencies on the database, the providers or the modelling stack, so
the scoring rules can be reasoned about and tested in isolation.
"""

from f1cp.domain.elimination import TitleContention, title_contention
from f1cp.domain.enums import IngestionMode, ProviderName, RunStatus, SeasonPhase, TableName
from f1cp.domain.phases import DEFAULT_PHASE_BINS, phase_for_progress
from f1cp.domain.scoring import (
    PointsSystem,
    max_points_per_round,
    points_for_win,
    points_system_for,
    second_best_points_per_round,
    sprint_points_for_win,
)

__all__ = [
    "DEFAULT_PHASE_BINS",
    "IngestionMode",
    "PointsSystem",
    "ProviderName",
    "RunStatus",
    "SeasonPhase",
    "TableName",
    "TitleContention",
    "max_points_per_round",
    "phase_for_progress",
    "points_for_win",
    "points_system_for",
    "second_best_points_per_round",
    "sprint_points_for_win",
    "title_contention",
]
