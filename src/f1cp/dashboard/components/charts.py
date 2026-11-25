"""Plotly figure builders shared by the dashboard's pages.

Every figure comes from one set of layout defaults, so no two charts disagree about a
margin, a font or a gridline. Magnitude over nominal categories gets no colour -- bar
length already carries the number, so those bars share a neutral and the accent marks the
one row worth pointing at. Identity charts draw from the fixed categorical order instead.
Chrome recedes: no background, no border (the card supplies both), no redundant grid.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from f1cp.dashboard.components.labels import percent
from f1cp.dashboard.components.palette import (
    ACCENT,
    FONT_STACK,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_NEUTRAL,
    SURFACE_RAISED,
    categorical_colors,
)

__all__ = [
    "driver_highlight_trajectory_chart",
    "feature_importance_chart",
    "metric_comparison_chart",
    "multi_driver_trajectory_chart",
    "probability_bar_chart",
]

_FONT = {"family": FONT_STACK, "size": 12, "color": INK_SECONDARY}

# Transparent, not surface-coloured: the figure inherits whichever panel it sits in, so a
# chart never shows a rectangle a shade off from its own card.
_LAYOUT_DEFAULTS: dict[str, object] = {
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": _FONT,
    "margin": {"l": 8, "r": 48, "t": 8, "b": 8},
    "hoverlabel": {
        "bgcolor": SURFACE_RAISED,
        "bordercolor": GRIDLINE,
        "font": {"family": FONT_STACK, "size": 12, "color": INK_PRIMARY},
    },
    "dragmode": False,
}

# `automargin` is not optional here: the figures carry a deliberately tight margin so
# charts sit flush inside their card, and without it Plotly clips long category labels
# (driver names, feature names) and the outermost tick labels to a couple of characters.
_VALUE_AXIS: dict[str, object] = {
    "gridcolor": GRIDLINE,
    "zeroline": False,
    "showline": False,
    "ticks": "",
    "automargin": True,
}

_CATEGORY_AXIS: dict[str, object] = {
    "showgrid": False,
    "zeroline": False,
    "showline": False,
    "ticks": "",
    "automargin": True,
}

_LEGEND: dict[str, object] = {
    "orientation": "h",
    "yanchor": "bottom",
    "y": 1.04,
    "x": 0,
    "font": {"family": FONT_STACK, "size": 12, "color": INK_SECONDARY},
    "bgcolor": "rgba(0,0,0,0)",
}


def _layout(**overrides: object) -> dict[str, object]:
    """The shared layout, with per-figure overrides merged over it.

    Merging rather than spreading ``_LAYOUT_DEFAULTS`` at each call site means a figure
    can override a default (a taller top margin to clear a legend) instead of colliding
    with it.
    """
    return {**_LAYOUT_DEFAULTS, **overrides}


def probability_bar_chart(
    frame: pd.DataFrame, *, value_col: str, label_col: str, top_n: int = 12
) -> go.Figure:
    """A ranked horizontal bar chart of one probability per driver.

    Every bar is the same neutral except the leader's, which takes the accent -- the
    ranking is the message, and exactly one row is the answer to "who is winning".

    Args:
        frame: One row per driver, any order; the chart ranks it itself.
        value_col: Column holding the probability (0-1).
        label_col: Column holding the display label (e.g. driver name).
        top_n: How many of the leading drivers to plot. A full grid is a long tail of
            near-zero bars that adds length without information -- the table beneath
            these charts is where the complete field belongs.

    Returns:
        A horizontal bar figure.
    """
    ranked = frame.sort_values(value_col, ascending=False).head(top_n)
    ordered = ranked.sort_values(value_col, ascending=True)
    colors = [SERIES_NEUTRAL] * len(ordered)
    if len(colors):
        colors[-1] = ACCENT

    figure = go.Figure(
        go.Bar(
            x=ordered[value_col],
            y=ordered[label_col],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            width=0.62,
            text=[percent(value) for value in ordered[value_col]],
            textposition="outside",
            textfont={"family": FONT_STACK, "size": 12, "color": INK_SECONDARY},
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:.2%} to win the title<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            xaxis={**_VALUE_AXIS, "tickformat": ".0%", "rangemode": "tozero"},
            yaxis={**_CATEGORY_AXIS, "tickfont": {"family": FONT_STACK, "size": 12}},
            showlegend=False,
            bargap=0.34,
            height=max(240, 34 * len(ordered) + 40),
        )
    )
    return figure


def feature_importance_chart(
    feature_names: list[str], importances: list[float], *, top_n: int = 15
) -> go.Figure:
    """A ranked horizontal bar chart of mean |SHAP value| per feature.

    Args:
        feature_names: Feature labels to show, most important first. Callers pass the
            readable labels, not column names.
        importances: Mean absolute SHAP value per feature, same order.
        top_n: How many of the leading features to plot.

    Returns:
        A horizontal bar figure. Features are nominal, so the strongest one takes the
        accent and the rest share one neutral -- same rule as the probability chart.
    """
    frame = pd.DataFrame({"feature": feature_names, "importance": importances}).head(top_n)
    ordered = frame.sort_values("importance", ascending=True)
    colors = [SERIES_NEUTRAL] * len(ordered)
    if len(colors):
        colors[-1] = ACCENT

    figure = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            width=0.6,
            hovertemplate="<b>%{y}</b><br>average influence %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            xaxis={
                **_VALUE_AXIS,
                "title": {
                    "text": "Average influence on the prediction",
                    "font": {"family": FONT_STACK, "size": 11, "color": INK_MUTED},
                },
                "rangemode": "tozero",
            },
            yaxis={**_CATEGORY_AXIS, "tickfont": {"family": FONT_STACK, "size": 12}},
            showlegend=False,
            bargap=0.36,
            height=max(280, 26 * len(ordered) + 60),
        )
    )
    return figure


def metric_comparison_chart(
    frame: pd.DataFrame, *, metric_col: str, category_col: str, lower_is_better: bool = True
) -> go.Figure:
    """A bar chart comparing one metric across the candidates of a training run.

    Args:
        frame: One row per model version/candidate.
        metric_col: Column holding the metric value.
        category_col: Column holding the category label (e.g. algorithm name).
        lower_is_better: Whether a smaller value is a better score, which decides which
            bar is the winner and therefore which one takes the accent.

    Returns:
        A vertical bar figure, sorted best-first.
    """
    ordered = frame.sort_values(metric_col, ascending=lower_is_better)
    colors = [SERIES_NEUTRAL] * len(ordered)
    if len(colors):
        colors[0] = ACCENT

    figure = go.Figure(
        go.Bar(
            x=ordered[category_col],
            y=ordered[metric_col],
            marker={"color": colors, "line": {"width": 0}},
            width=0.56,
            hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(
            yaxis={
                **_VALUE_AXIS,
                # The metric is chosen from a control above the chart, so naming it on the
                # axis is what keeps a screenshot of the chart alone self-explanatory.
                "title": {
                    "text": metric_col.replace("_", " "),
                    "font": {"family": FONT_STACK, "size": 11, "color": INK_MUTED},
                },
                "rangemode": "tozero",
            },
            xaxis={**_CATEGORY_AXIS, "tickfont": {"family": FONT_STACK, "size": 12}},
            showlegend=False,
            bargap=0.42,
            height=320,
        )
    )
    return figure


def driver_highlight_trajectory_chart(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    label_col: str,
    highlight_id: str,
) -> go.Figure:
    """A probability trajectory with one driver picked out against the rest of the field.

    Every other driver is drawn as a thin, muted, legend-free line -- this is what makes a
    single-driver view of a twenty-plus-driver season legible without inventing extra
    hues: the highlighted driver is the only one that needs a colour at all, and here that
    colour is the accent, because they are what the page is about.

    Args:
        frame: Long-format rows of ``(x_col, y_col, group_col, label_col)``.
        x_col: Column holding the round number.
        y_col: Column holding the probability.
        group_col: Column identifying each line (e.g. driver id).
        label_col: Column holding the display label for the highlighted line.
        highlight_id: The ``group_col`` value to highlight.

    Returns:
        A line chart figure.
    """
    figure = go.Figure()
    for group_id, group_frame in frame.groupby(group_col, sort=False):
        is_highlighted = group_id == highlight_id
        if is_highlighted:
            continue
        ordered = group_frame.sort_values(x_col)
        figure.add_trace(
            go.Scatter(
                x=ordered[x_col],
                y=ordered[y_col],
                mode="lines",
                name=str(group_id),
                line={"color": INK_MUTED, "width": 1},
                opacity=0.28,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    highlighted = frame[frame[group_col] == highlight_id].sort_values(x_col)
    if len(highlighted):
        figure.add_trace(
            go.Scatter(
                x=highlighted[x_col],
                y=highlighted[y_col],
                mode="lines+markers",
                name=str(highlighted[label_col].iloc[0]),
                line={"color": ACCENT, "width": 2.5},
                # A surface-coloured ring keeps markers legible where the line crosses
                # the muted field behind it.
                marker={
                    "size": 8,
                    "color": ACCENT,
                    "line": {"width": 2, "color": SURFACE_RAISED},
                },
                showlegend=False,
                hovertemplate="Round %{x} &middot; %{y:.1%}<extra></extra>",
            )
        )

    figure.update_layout(
        **_layout(
            xaxis={**_CATEGORY_AXIS, "title": {"text": "Round", "font": {"size": 11}}, "tickformat": "d"},
            yaxis={**_VALUE_AXIS, "tickformat": ".0%", "rangemode": "tozero"},
            hovermode="x unified",
            height=400,
        )
    )
    return figure


def multi_driver_trajectory_chart(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    label_col: str,
) -> go.Figure:
    """A probability trajectory for a small set of drivers, each with its own colour.

    Args:
        frame: Long-format rows of ``(x_col, y_col, group_col, label_col)``. Callers
            should cap the number of distinct ``group_col`` values at eight before
            calling this -- past that, colours are no longer validated as
            distinguishable.
        x_col: Column holding the round number.
        y_col: Column holding the probability.
        group_col: Column identifying each line (e.g. driver id).
        label_col: Column holding each line's legend label.

    Returns:
        A line chart figure with a legend. Colours are assigned by first appearance in
        ``frame`` rather than by rank, so adding or removing a driver never repaints the
        ones that stayed.
    """
    groups = list(dict.fromkeys(frame[group_col]))
    colors = dict(zip(groups, categorical_colors(len(groups)), strict=True))

    figure = go.Figure()
    for group_id, group_frame in frame.groupby(group_col, sort=False):
        ordered = group_frame.sort_values(x_col)
        figure.add_trace(
            go.Scatter(
                x=ordered[x_col],
                y=ordered[y_col],
                mode="lines+markers",
                name=str(ordered[label_col].iloc[0]),
                line={"color": colors[group_id], "width": 2},
                marker={
                    "size": 6,
                    "line": {"width": 1.5, "color": SURFACE_RAISED},
                },
                hovertemplate="%{fullData.name}: %{y:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        **_layout(
            xaxis={**_CATEGORY_AXIS, "title": {"text": "Round", "font": {"size": 11}}, "tickformat": "d"},
            yaxis={**_VALUE_AXIS, "tickformat": ".0%", "rangemode": "tozero"},
            hovermode="x unified",
            legend=_LEGEND,
            margin={"l": 8, "r": 48, "t": 40, "b": 8},
            height=420,
        )
    )
    return figure
