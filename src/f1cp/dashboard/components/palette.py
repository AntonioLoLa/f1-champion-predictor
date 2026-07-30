"""The dashboard's design tokens: one source of truth for every colour in the app.

Dark-first and deliberately near-monochrome. Formula 1's red is an *accent*, not a theme:
it marks the one thing that matters on a screen and nothing else. Colour does three jobs,
each with one rule:

* **Emphasis** -- ``ACCENT``, at most one mark per view.
* **Identity** -- ``CATEGORICAL``, assigned in fixed order and never cycled, so a series
  keeps its colour when the selection changes.
* **State** -- ``POSITIVE`` / ``NEGATIVE``, for signed deltas only, always with a sign.

Magnitude over nominal categories uses no colour at all: bar length already carries the
number, so those charts take ``SERIES_NEUTRAL`` and spend the accent on the one row worth
pointing at.

``.streamlit/config.toml`` duplicates the surface, ink and accent values for Streamlit's
own chrome -- it is TOML, so keep the two in sync by hand.
"""

from __future__ import annotations

__all__ = [
    "ACCENT",
    "ACCENT_MUTED",
    "ACCENT_SOFT",
    "BORDER",
    "BORDER_STRONG",
    "CATEGORICAL",
    "FONT_STACK",
    "GRIDLINE",
    "INK_MUTED",
    "INK_PRIMARY",
    "INK_SECONDARY",
    "NEGATIVE",
    "POSITIVE",
    "SERIES_NEUTRAL",
    "SURFACE",
    "SURFACE_RAISED",
    "SURFACE_SUNKEN",
    "categorical_colors",
]


SURFACE = "#0f1117"
"""Page background. Every contrast figure in this module is measured against it."""

SURFACE_RAISED = "#161923"
"""Cards and panels -- one step up from the page, separated by light, not by a shadow."""

SURFACE_SUNKEN = "#0b0d12"
"""Table headers and inset wells -- one step down, to push structure behind content."""

BORDER = "#232733"
BORDER_STRONG = "#2e3442"
GRIDLINE = "#1c2029"
"""Chart gridlines. Deliberately dimmer than :data:`BORDER`: a grid is a reading aid."""


INK_PRIMARY = "#f2f3f5"
"""Off-white rather than pure white -- less glare against a near-black surface."""

INK_SECONDARY = "#9ba1ae"
INK_MUTED = "#6b7280"


ACCENT = "#e10600"
"""Formula 1 red. Reserved for emphasis; never a series colour, never decoration."""

ACCENT_MUTED = "#8f1410"
ACCENT_SOFT = "rgba(225, 6, 0, 0.12)"


SERIES_NEUTRAL = "#8b93a7"
"""Single-series magnitude marks (probability bars, SHAP bars).

A neutral slate, not a hue: these bars encode magnitude through length, so a hue here
would claim an identity that does not exist. The chroma floor that governs
:data:`CATEGORICAL` does not apply -- that check exists to keep eight *identities*
distinguishable, and this mark has no identity to distinguish. It clears 3:1 against
:data:`SURFACE`, which is the check that does apply.
"""

CATEGORICAL: tuple[str, ...] = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
)
"""Identity hues, in fixed assignment order.

Validated as a set against :data:`SURFACE`: every slot sits inside the dark lightness
band, clears the chroma floor and 3:1 contrast, and the worst adjacent pair holds ΔE 8.4
under simulated protanopia (target 8) and 19.3 under normal vision (floor 15). The
*order* is the colourblind-safety mechanism, not a preference -- do not reorder it, and
do not extend it, without re-running the validation.
"""

POSITIVE = "#3fb950"
NEGATIVE = "#f85149"
"""Signed-delta colours. Distinct from :data:`ACCENT` on purpose: a falling number and
"the thing to look at" are different ideas and must not share a colour."""


FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif'
)
"""Matches ``.streamlit/config.toml`` so Plotly text and page text are the same face."""


def categorical_colors(n: int) -> list[str]:
    """The first ``n`` identity colours, folding anything past the palette into gray.

    Args:
        n: Number of distinct series needed.

    Returns:
        ``n`` hex colours. Beyond the eight validated slots, additional series fall back
        to a neutral gray rather than an unvalidated, possibly colliding hue -- callers
        with many series should cap what they plot (top-N, "Other") rather than rely on
        this.
    """
    colors = list(CATEGORICAL[:n])
    if n > len(CATEGORICAL):
        colors.extend([INK_MUTED] * (n - len(CATEGORICAL)))
    return colors
