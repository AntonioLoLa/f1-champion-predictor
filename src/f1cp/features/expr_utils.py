"""Small polars expression helpers shared across feature blocks.

Kept separate from any one block because the same pattern -- a null-safe cumulative mean
over a partition -- recurs in qualifying, circuit and constructor features.
"""

from __future__ import annotations

import polars as pl

__all__ = ["cum_mean"]


def cum_mean(expr: pl.Expr) -> pl.Expr:
    """Null-safe cumulative mean of an expression.

    ``Expr.cum_sum`` alone is not enough: it emits ``null`` at any row where the current
    value is null rather than carrying the running total forward, and there is no built-in
    cumulative-mean expression. Filling nulls with zero for the running sum and dividing by
    a running count of non-null values gives the intended "average of everything seen so
    far, ignoring gaps" semantics.

    Args:
        expr: The expression to average cumulatively. Typically already ``shift(1)``ed by
            the caller so the current row is excluded.

    Returns:
        An expression evaluating to the cumulative mean, ``null`` where no non-null value
        has been seen yet. Apply ``.over(group)`` to the result to partition it.
    """
    running_sum = expr.fill_null(0).cum_sum()
    running_count = expr.is_not_null().cast(pl.Int32).cum_sum()
    return pl.when(running_count > 0).then(running_sum / running_count).otherwise(None)
