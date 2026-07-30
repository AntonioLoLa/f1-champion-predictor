"""Plotly figures for SHAP and permutation importance.

Native Plotly horizontal bar charts rather than ``shap.plots.*`` (which renders with
matplotlib) -- the project is Plotly-only end to end, so every figure in the report shares
one rendering stack, one theme, and one JSON/PNG export path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from f1cp.explain.permutation import PermutationResult
    from f1cp.explain.shap_explainer import ShapResult

__all__ = ["permutation_importance_figure", "shap_importance_figure"]


def shap_importance_figure(result: ShapResult, *, top_n: int) -> go.Figure:
    """Horizontal bar chart of mean |SHAP value|, most important feature on top.

    Args:
        result: Output of :func:`f1cp.explain.shap_explainer.explain`.
        top_n: Number of top features to display.

    Returns:
        A horizontal bar chart of the top ``top_n`` features.
    """
    names = result.feature_names[:top_n]
    values = result.importances[:top_n]
    figure = go.Figure(data=go.Bar(x=values, y=names, orientation="h"))
    figure.update_layout(
        title="SHAP feature importance (mean |SHAP value|)",
        xaxis_title="Mean |SHAP value|",
        yaxis={"autorange": "reversed"},
    )
    return figure


def permutation_importance_figure(result: PermutationResult, *, top_n: int) -> go.Figure:
    """Horizontal bar chart of permutation importance with error bars.

    Args:
        result: Output of :func:`f1cp.explain.permutation.explain`.
        top_n: Number of top features to display.

    Returns:
        A horizontal bar chart of the top ``top_n`` features, error bars showing the
        standard deviation across repeats.
    """
    names = result.feature_names[:top_n]
    values = result.importances_mean[:top_n]
    errors = result.importances_std[:top_n]
    figure = go.Figure(
        data=go.Bar(
            x=values,
            y=names,
            orientation="h",
            error_x={"type": "data", "array": errors},
        )
    )
    figure.update_layout(
        title="Permutation importance (mean drop in negative log loss)",
        xaxis_title="Importance",
        yaxis={"autorange": "reversed"},
    )
    return figure
