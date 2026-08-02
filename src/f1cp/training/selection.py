"""Model selection: choosing one winner from the zoo's trained candidates.

Accuracy would pick "always predict not champion" (see :mod:`f1cp.training.metrics`), so
the primary metric and its direction come from ``training.selection``.

The harder problem is not *which* metric but *how much of a difference counts*. Each
candidate's headline number is a mean over walk-forward folds -- thirty seasons, one per
fold -- and the spread between seasons is large, because some championships are decided by
round twelve and some go to the final corner. Taking a bare ``min()`` over those means
treats a gap of 0.01 exactly like a gap of 1.0, and on this project's real data that was
not a hypothetical concern: the top three candidates sat inside 0.14 with a standard error
of 0.09 on the leader, and the runner-up was in fact *better* on 18 of the 30 individual
seasons. Declaring a winner from that is reporting a coin flip as a result.

So selection applies the standard **one-standard-error rule**: compute the standard error
of the best candidate's fold mean, take every candidate whose mean falls within one of
those of the best, and choose among that set by an explicit preference order rather than by
a difference the data cannot resolve. Defaulting that order to "simplest model first" is
the usual reading of the rule -- prefer the least complex model that the evidence cannot
distinguish from the best -- and it makes the choice reproducible and arguable instead of
incidental.

The rule usually confirms the bare minimum rather than overturning it. That is the point:
what changes is that the run now records *why*, and which candidates it could not tell
apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from f1cp.config.settings import SelectionConfig

__all__ = ["Scored", "SelectionResult", "select_best"]


class Scored(Protocol):
    """Anything with a name, aggregated metrics and per-fold metrics can be selected."""

    @property
    def name(self) -> str:
        """The candidate's identifier, e.g. a model zoo entry name."""
        ...

    @property
    def metrics(self) -> dict[str, float | None]:
        """Aggregated metric values, keyed by metric name."""
        ...

    @property
    def fold_metrics(self) -> list[dict[str, float | None]]:
        """One dict of metric values per walk-forward fold, in fold order."""
        ...


@dataclass(frozen=True, slots=True)
class SelectionResult[T: Scored]:
    """The outcome of a selection, and enough context to argue with it.

    Attributes:
        winner: The chosen candidate.
        primary_metric: Metric the choice was made on.
        best_value: The best fold-mean observed.
        standard_error: Standard error of that mean across folds, or ``None`` when there
            were too few folds to estimate one.
        threshold: The value a candidate had to reach to count as indistinguishable from
            the best, or ``None`` when the rule did not apply.
        tied: Names of every candidate within one standard error of the best, in metric
            order. A single entry means the winner was resolved outright.
        rule: ``"one_standard_error"`` when the tie set decided it, ``"best_mean"`` when
            the rule was disabled or could not be computed.
    """

    winner: T
    primary_metric: str
    best_value: float | None
    standard_error: float | None
    threshold: float | None
    tied: list[str] = field(default_factory=list)
    rule: str = "best_mean"

    def summary(self) -> dict[str, object]:
        """A JSON-serialisable record of the decision, for the training run's params."""
        return {
            "winner": self.winner.name,
            "primary_metric": self.primary_metric,
            "best_value": self.best_value,
            "standard_error": self.standard_error,
            "threshold": self.threshold,
            "tied": list(self.tied),
            "rule": self.rule,
        }


def _fold_values(candidate: Scored, metric: str) -> list[float]:
    """The candidate's per-fold values for one metric, skipping folds that had none."""
    return [value for fold in candidate.fold_metrics if (value := fold.get(metric)) is not None]


def _standard_error(values: Sequence[float]) -> float | None:
    """Standard error of the mean, or ``None`` below two observations."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def select_best[T: Scored](candidates: Sequence[T], config: SelectionConfig) -> SelectionResult[T]:
    """Pick a winner, preferring simplicity over differences the folds cannot resolve.

    Args:
        candidates: Every trained candidate, with its aggregated and per-fold CV metrics.
        config: ``training.selection`` settings: which metric decides, which direction is
            better, whether to apply the one-standard-error rule, the preference order used
            to break a tie, and the metrics used to break what the order leaves open.

    Returns:
        The winning candidate together with the tie set and the numbers behind the choice.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        msg = "Cannot select a best candidate from an empty list"
        raise ValueError(msg)

    sign = 1.0 if config.direction == "minimize" else -1.0

    def oriented(candidate: T) -> float:
        """The primary metric, signed so that lower is always better."""
        value = candidate.metrics.get(config.primary_metric)
        return math.inf if value is None else sign * value

    def tie_break_key(candidate: T) -> tuple[float, ...]:
        """Preference order first, then the configured error metrics, then the metric."""
        try:
            preference = config.preference_order.index(candidate.name)
        except ValueError:
            preference = len(config.preference_order)
        key = [float(preference)]
        for tie_breaker in config.tie_breakers:
            value = candidate.metrics.get(tie_breaker)
            key.append(math.inf if value is None else value)
        key.append(oriented(candidate))
        return tuple(key)

    ranked = sorted(candidates, key=oriented)
    leader = ranked[0]
    best_oriented = oriented(leader)
    best_value = leader.metrics.get(config.primary_metric)

    standard_error = _standard_error(_fold_values(leader, config.primary_metric))
    if not config.one_standard_error or standard_error is None or math.isinf(best_oriented):
        return SelectionResult(
            winner=leader,
            primary_metric=config.primary_metric,
            best_value=best_value,
            standard_error=standard_error,
            threshold=None,
            tied=[leader.name],
            rule="best_mean",
        )

    limit = best_oriented + standard_error
    tied = [candidate for candidate in ranked if oriented(candidate) <= limit]
    return SelectionResult(
        winner=min(tied, key=tie_break_key),
        primary_metric=config.primary_metric,
        best_value=best_value,
        standard_error=standard_error,
        threshold=sign * limit,
        tied=[candidate.name for candidate in tied],
        rule="one_standard_error",
    )
