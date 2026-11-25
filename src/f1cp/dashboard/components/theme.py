"""The dashboard's visual system: one stylesheet, and the primitives built on top of it.

Two rules keep it maintainable. Every colour comes from
:mod:`f1cp.dashboard.components.palette`, so the chrome and the charts change together.
And text hierarchy uses native Streamlit elements rather than our own markup, so a page
stays legible even if the stylesheet never lands -- the CSS is polish, not structure.

Nothing here reads or transforms data; it decides only how already-computed values look.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Literal

import streamlit as st

from f1cp.dashboard.components import palette as p

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "BoardRow",
    "Stat",
    "card",
    "empty_state",
    "empty_state_html",
    "inject_theme",
    "leaderboard",
    "leaderboard_html",
    "page_header",
    "section",
    "section_html",
    "stat_grid",
    "stat_grid_html",
    "stylesheet",
]

Tone = Literal["neutral", "positive", "negative", "accent"]

# How a stat's supporting line is coloured. Streamlit reads a leading "+"/"-" and colours
# accordingly under "normal"; "off" keeps the line neutral, which is what a plain
# descriptive detail (a team name, a unit) wants.
_DELTA_COLOURS: dict[Tone, Literal["normal", "off"]] = {
    "neutral": "off",
    "accent": "off",
    "positive": "normal",
    "negative": "normal",
}


@dataclass(frozen=True, slots=True)
class Stat:
    """One headline number in a :func:`stat_grid`.

    Attributes:
        label: What the number measures, in sentence case.
        value: The formatted number. Formatting is the caller's job -- this module never
            rounds or converts, so the value shown is exactly the value passed.
        detail: Optional supporting line beneath the value (a comparison, a unit, a name).
        tone: Colours ``detail`` only. ``accent`` additionally marks the whole card as the
            one that matters; use it at most once per grid.
    """

    label: str
    value: str
    detail: str | None = None
    tone: Tone = "neutral"


@dataclass(frozen=True, slots=True)
class BoardRow:
    """One line of a :func:`leaderboard`.

    Attributes:
        position: Rank label, e.g. ``"1"``.
        name: Primary label, e.g. the driver's name.
        secondary: Optional smaller line beneath the name, e.g. their constructor.
        value: The formatted number shown at the right.
        share: ``0``-``1``, drawn as a hairline meter under the row. Callers pass the
            same quantity ``value`` displays, so the two never disagree.
    """

    position: str
    name: str
    value: str
    secondary: str | None = None
    share: float = 0.0


def _stylesheet() -> str:
    """Build the global stylesheet from the palette tokens."""
    return f"""
    <style>
      :root {{
        --f1-surface: {p.SURFACE};
        --f1-raised: {p.SURFACE_RAISED};
        --f1-sunken: {p.SURFACE_SUNKEN};
        --f1-border: {p.BORDER};
        --f1-border-strong: {p.BORDER_STRONG};
        --f1-ink: {p.INK_PRIMARY};
        --f1-ink-2: {p.INK_SECONDARY};
        --f1-ink-3: {p.INK_MUTED};
        --f1-accent: {p.ACCENT};
        --f1-accent-soft: {p.ACCENT_SOFT};
        --f1-positive: {p.POSITIVE};
        --f1-negative: {p.NEGATIVE};
        --f1-radius: 10px;
        --f1-radius-sm: 7px;
        /* One easing and one duration for every transition in the app. */
        --f1-ease: cubic-bezier(0.22, 0.61, 0.36, 1);
        --f1-fast: 140ms;
      }}

      /* Canvas */

      .stApp {{ background: var(--f1-surface); }}

      .stMainBlockContainer {{
        padding-bottom: 6rem;
        max-width: 1320px;
      }}

      /* Tabular figures everywhere a number appears, so digits line up in columns
         and a changing value does not shift the ones beside it. */
      .stApp, .stApp input, .stApp button {{
        font-feature-settings: "tnum" 1, "cv05" 1;
        -webkit-font-smoothing: antialiased;
      }}

      /* Typography */

      .stApp h1, .stApp h2, .stApp h3 {{
        color: var(--f1-ink);
        letter-spacing: -0.02em;
        font-weight: 600;
      }}

      .stMarkdown p, .stMarkdown li {{ color: var(--f1-ink-2); }}

      /* Header */

      .f1-header__eyebrow {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.6875rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--f1-ink-3);
        margin-bottom: 0.6rem;
      }}
      /* The one piece of brand furniture in the app: a short red rule, not a logo. */
      .f1-header__rule {{
        width: 20px;
        height: 2px;
        background: var(--f1-accent);
        border-radius: 1px;
      }}
      .f1-header__title {{
        font-size: 1.9rem;
        font-weight: 600;
        line-height: 1.15;
        letter-spacing: -0.03em;
        color: var(--f1-ink);
        margin: 0;
      }}
      .f1-header__subtitle {{
        font-size: 0.9375rem;
        line-height: 1.5;
        color: var(--f1-ink-2);
        margin: 0.5rem 0 0;
        max-width: 68ch;
      }}
      .f1-header__meta {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        padding-bottom: 0.2rem;
      }}

      /* Chips */

      .f1-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.65rem;
        border: 1px solid var(--f1-border);
        border-radius: 999px;
        background: var(--f1-raised);
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--f1-ink-2);
        white-space: nowrap;
      }}
      .f1-chip__dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--f1-ink-3);
        flex: none;
      }}
      .f1-chip--ok .f1-chip__dot {{ background: var(--f1-positive); }}
      .f1-chip--bad .f1-chip__dot {{ background: var(--f1-negative); }}
      .f1-chip--accent {{
        border-color: color-mix(in srgb, var(--f1-accent) 45%, transparent);
        background: var(--f1-accent-soft);
        color: var(--f1-ink);
      }}
      .f1-chip--accent .f1-chip__dot {{ background: var(--f1-accent); }}

      /* Sections */

      /* A hairline above each section, not just space: with several stacked sections a
         reader needs to see where one ends, and margin alone reads as "same block". */
      .f1-section {{
        margin: 3rem 0 1.25rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--f1-border);
      }}
      .f1-section__title {{
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.3;
        letter-spacing: -0.02em;
        color: var(--f1-ink);
        margin: 0;
      }}
      .f1-section__note {{
        font-size: 0.875rem;
        line-height: 1.6;
        color: var(--f1-ink-3);
        margin: 0.5rem 0 0;
        max-width: 78ch;
      }}

      /* Stats */

      .f1-stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 0.75rem;
        margin: 0.25rem 0 0.5rem;
      }}
      .f1-stat {{
        position: relative;
        overflow: hidden;
        padding: 1.05rem 1.15rem;
        border: 1px solid var(--f1-border);
        border-radius: var(--f1-radius);
        background: var(--f1-raised);
        transition: border-color var(--f1-fast) var(--f1-ease);
      }}
      .f1-stat:hover {{ border-color: var(--f1-border-strong); }}
      /* The accent card is marked by a hairline on its edge, not a filled background:
         emphasis without turning a data card into a banner. */
      .f1-stat--accent::before {{
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 2px;
        background: var(--f1-accent);
      }}
      .f1-stat__label {{
        font-size: 0.6875rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--f1-ink-3);
      }}
      .f1-stat__value {{
        font-size: 1.75rem;
        font-weight: 600;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: var(--f1-ink);
        margin-top: 0.5rem;
      }}
      .f1-stat__detail {{
        font-size: 0.8125rem;
        color: var(--f1-ink-2);
        margin-top: 0.4rem;
      }}
      .f1-stat__detail--positive {{ color: var(--f1-positive); }}
      .f1-stat__detail--negative {{ color: var(--f1-negative); }}
      .f1-stat__detail--accent {{ color: var(--f1-ink); }}

      /* Cards */

      .f1-card {{
        border: 1px solid var(--f1-border);
        border-radius: var(--f1-radius);
        background: var(--f1-raised);
        padding: 1.25rem 1.35rem;
      }}
      .f1-card__title {{
        font-size: 0.8125rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--f1-ink-3);
        margin: 0 0 0.9rem;
      }}

      /* Empty state */

      .f1-empty {{
        border: 1px dashed var(--f1-border-strong);
        border-radius: var(--f1-radius);
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: var(--f1-ink-2);
        background: color-mix(in srgb, var(--f1-raised) 55%, transparent);
      }}
      .f1-empty__title {{
        color: var(--f1-ink);
        font-weight: 600;
        font-size: 0.9375rem;
        margin-bottom: 0.35rem;
      }}
      .f1-empty__body {{ font-size: 0.875rem; max-width: 46ch; margin: 0 auto; }}

      /* Leaderboard */

      .f1-board {{ display: flex; flex-direction: column; }}
      .f1-board__row {{
        display: grid;
        grid-template-columns: 2rem 1fr auto;
        align-items: center;
        gap: 0.9rem;
        padding: 0.6rem 0.25rem 0.7rem;
        border-bottom: 1px solid var(--f1-border);
      }}
      .f1-board__row:last-child {{ border-bottom: none; }}
      .f1-board__pos {{
        font-size: 0.8125rem;
        font-weight: 600;
        color: var(--f1-ink-3);
        text-align: right;
      }}
      .f1-board__row--lead .f1-board__pos {{ color: var(--f1-accent); }}
      .f1-board__name {{
        font-size: 0.9375rem;
        color: var(--f1-ink);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .f1-board__team {{
        display: block;
        font-size: 0.75rem;
        color: var(--f1-ink-3);
        margin-top: 0.1rem;
      }}
      .f1-board__value {{
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--f1-ink);
        font-variant-numeric: tabular-nums;
      }}
      /* The probability bar sits under the name as a hairline: a second read of the
         same number for scanning, without adding a chart. */
      .f1-board__meter {{
        grid-column: 2 / 4;
        height: 2px;
        border-radius: 1px;
        background: var(--f1-border);
        overflow: hidden;
        margin-top: 0.3rem;
      }}
      .f1-board__meter span {{
        display: block;
        height: 100%;
        border-radius: 1px;
        background: {p.SERIES_NEUTRAL};
      }}
      .f1-board__row--lead .f1-board__meter span {{ background: var(--f1-accent); }}

      /* Streamlit */

      /* Sidebar: a quieter plane than the page, separated by a hairline. */
      section[data-testid="stSidebar"] {{
        background: var(--f1-sunken);
        border-right: 1px solid var(--f1-border);
      }}
      section[data-testid="stSidebar"] .stMainBlockContainer {{ padding-top: 1.5rem; }}

      div[data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: var(--f1-accent-soft) !important;
        color: var(--f1-ink) !important;
      }}

      /* Tables: header sits back, rows lift on hover, hairline separators only. */
      div[data-testid="stDataFrame"] {{
        border: 1px solid var(--f1-border);
        border-radius: var(--f1-radius);
        overflow: hidden;
      }}

      /* Buttons: flat, hairline-bordered, one accent action per view. */
      .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
        border-radius: var(--f1-radius-sm);
        border: 1px solid var(--f1-border-strong);
        font-weight: 500;
        transition: background var(--f1-fast) var(--f1-ease),
                    border-color var(--f1-fast) var(--f1-ease),
                    transform var(--f1-fast) var(--f1-ease);
      }}
      .stButton > button:hover, .stFormSubmitButton > button:hover {{
        border-color: var(--f1-ink-3);
        transform: translateY(-1px);
      }}
      .stButton > button:active, .stFormSubmitButton > button:active {{
        transform: translateY(0);
      }}

      /* Inputs: recessive until focused. */
      .stSelectbox div[data-baseweb="select"] > div,
      .stTextInput input, .stNumberInput input, .stMultiSelect div[data-baseweb="select"] > div {{
        background: var(--f1-raised);
        border-color: var(--f1-border);
        border-radius: var(--f1-radius-sm);
      }}

      div[data-testid="stExpander"] {{
        border: 1px solid var(--f1-border);
        border-radius: var(--f1-radius);
        background: color-mix(in srgb, var(--f1-raised) 60%, transparent);
      }}

      /* Streamlit's own horizontal rule is heavier than this design wants. */
      hr, div[data-testid="stDivider"] hr {{ border-color: var(--f1-border); }}

      /* Animation */

      /* One entrance, applied once to the page body. Deliberately short and subtle:
         enough to soften a rerun, not enough to notice twice. */
      @keyframes f1-rise {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to   {{ opacity: 1; transform: none; }}
      }}
      .stMainBlockContainer > div {{
        animation: f1-rise 260ms var(--f1-ease) both;
      }}

      @media (prefers-reduced-motion: reduce) {{
        .stMainBlockContainer > div {{ animation: none; }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{ transform: none; }}
      }}

      /* Responsive */

      @media (max-width: 900px) {{
        .f1-header {{ align-items: flex-start; }}
        .f1-header__title {{ font-size: 1.55rem; }}
        .f1-stats {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
        .f1-stat__value {{ font-size: 1.5rem; }}
      }}
    </style>
    """


def inject_theme() -> None:
    """Emit the global stylesheet, on every run.

    Streamlit drops anything a rerun does not re-emit, so injecting this once behind a
    session flag leaves the app unstyled from the user's first click onwards. Emitting it
    twice in one run is harmless, which is what lets the sidebar helpers call it without
    knowing whether the page header ran first.

    Goes through ``st.html``, the API Streamlit documents for raw CSS. It is polish over a
    layout that already reads without it -- see :func:`section` and :func:`stat_grid`.
    """
    st.html(_stylesheet())


# The builders below are pure -- markup in, string out -- so they can be unit-tested and
# rendered outside a running Streamlit session.


def _chip(label: str, *, variant: str = "") -> str:
    modifier = f" f1-chip--{variant}" if variant else ""
    return (
        f'<span class="f1-chip{modifier}"><span class="f1-chip__dot"></span>'
        f"{escape(label)}</span>"
    )


def section_html(title: str, note: str | None = None) -> str:
    """A section heading's markup. See :func:`section`."""
    note_html = f'<p class="f1-section__note">{escape(note)}</p>' if note else ""
    return (
        f'<div class="f1-section"><p class="f1-section__title">{escape(title)}</p>'
        f"{note_html}</div>"
    )


def stat_grid_html(stats: Sequence[Stat]) -> str:
    """The headline-number grid's markup. See :func:`stat_grid`."""
    cards = []
    for stat in stats:
        accent = " f1-stat--accent" if stat.tone == "accent" else ""
        detail = (
            f'<div class="f1-stat__detail f1-stat__detail--{stat.tone}">'
            f"{escape(stat.detail)}</div>"
            if stat.detail
            else ""
        )
        cards.append(
            f'<div class="f1-stat{accent}">'
            f'<div class="f1-stat__label">{escape(stat.label)}</div>'
            f'<div class="f1-stat__value">{escape(stat.value)}</div>'
            f"{detail}</div>"
        )
    return f'<div class="f1-stats">{"".join(cards)}</div>'


def leaderboard_html(rows: Sequence[BoardRow], *, highlight_first: bool = True) -> str:
    """The ranked-rows markup. See :func:`leaderboard`."""
    rendered = []
    for index, row in enumerate(rows):
        lead = " f1-board__row--lead" if highlight_first and index == 0 else ""
        secondary = (
            f'<span class="f1-board__team">{escape(row.secondary)}</span>' if row.secondary else ""
        )
        width = f"{max(0.0, min(row.share, 1.0)) * 100:.1f}%"
        rendered.append(
            f'<div class="f1-board__row{lead}">'
            f'<div class="f1-board__pos">{escape(row.position)}</div>'
            f'<div class="f1-board__name">{escape(row.name)}{secondary}</div>'
            f'<div class="f1-board__value">{escape(row.value)}</div>'
            f'<div class="f1-board__meter"><span style="width:{width}"></span></div>'
            f"</div>"
        )
    return f'<div class="f1-board">{"".join(rendered)}</div>'


def empty_state_html(title: str, body: str) -> str:
    """The empty-state panel's markup. See :func:`empty_state`."""
    return (
        f'<div class="f1-empty"><div class="f1-empty__title">{escape(title)}</div>'
        f'<div class="f1-empty__body">{escape(body)}</div></div>'
    )


def stylesheet() -> str:
    """The app's global stylesheet, ``<style>`` tag included."""
    return _stylesheet()


def page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str = "F1 Championship Predictor",
    meta: Sequence[tuple[str, str]] = (),
) -> None:
    """Render the standard page header.

    Every page opens with exactly this block, which is what makes eight separate screens
    read as one product.

    Args:
        title: The page's own name.
        subtitle: One sentence on what the page answers.
        eyebrow: Small uppercase line above the title. Defaults to the product name, so
            the app identifies itself without a logo or a navbar full of icons.
        meta: ``(label, variant)`` chips shown at the right of the header -- context that
            qualifies everything below it, such as the active model or the round the
            numbers describe. ``variant`` is ``""``, ``"ok"``, ``"bad"`` or ``"accent"``.
    """
    inject_theme()
    # Same reasoning as `section`: the title and its one-line explanation are native
    # elements so they keep a real hierarchy without the stylesheet, and only the chip row
    # -- which has no native equivalent -- is our own markup.
    st.markdown(
        f'<div class="f1-header__eyebrow"><span class="f1-header__rule"></span>'
        f"{escape(eyebrow)}</div>",
        unsafe_allow_html=True,
    )
    st.title(title, anchor=False)
    st.caption(subtitle)
    if meta:
        chips = "".join(_chip(label, variant=variant) for label, variant in meta)
        st.markdown(f'<div class="f1-header__meta">{chips}</div>', unsafe_allow_html=True)
    st.divider()


def section(title: str, note: str | None = None) -> None:
    """Render a section heading that separates content without shouting.

    Args:
        title: The section's name. Say what the reader will find, not what kind of
            object it is -- "Who can still win it" beats "Results".
        note: Optional one-line explanation beneath it.
    """
    # Native elements, not our own markup: Streamlit already renders a subheader and a
    # caption at visibly different sizes and colours, so the title/description distinction
    # holds even if the injected stylesheet never lands. The CSS above restyles these; it
    # is not what makes them legible.
    st.subheader(title, anchor=False)
    if note:
        st.caption(note)


def stat_grid(stats: Sequence[Stat]) -> None:
    """Render headline numbers as a responsive row of cards.

    Args:
        stats: The numbers to show, in reading order. Values are rendered exactly as
            given. Four or fewer keeps each card wide enough to read at a glance.
    """
    if not stats:
        return
    columns = st.columns(len(stats), gap="small")
    for column, stat in zip(columns, stats, strict=True):
        with column, st.container(border=True):
            # `st.metric` is the one native widget that already renders a small muted
            # label, a large value and a coloured supporting line -- the exact hierarchy
            # this grid needs, without depending on the injected stylesheet to supply it.
            st.metric(
                stat.label,
                stat.value,
                delta=stat.detail,
                delta_color=_DELTA_COLOURS[stat.tone],
                border=False,
            )


def leaderboard(rows: Sequence[BoardRow], *, highlight_first: bool = True) -> None:
    """Render a ranked list as hairline-separated rows rather than a table.

    A short ranking is not tabular data -- there is one number per row and the order *is*
    the message. Rows read faster than a grid, and the meter under each gives the
    magnitude a second, scannable encoding without spending a chart on it.

    Args:
        rows: The ranking, already in order.
        highlight_first: Whether the leading row takes the accent. Turn this off where
            the leader is not meaningfully "the answer".
    """
    if not rows:
        return
    st.markdown(leaderboard_html(rows, highlight_first=highlight_first), unsafe_allow_html=True)


@contextmanager
def card(title: str | None = None) -> Iterator[None]:
    """Wrap Streamlit widgets in a bordered panel.

    Uses ``st.container(border=True)`` rather than raw HTML: the content is real
    Streamlit output (a chart, a table), so the container has to be a real element the
    framework renders into, not a string.

    Args:
        title: Optional small uppercase label at the top of the panel.

    Yields:
        Nothing; write Streamlit output inside the ``with`` block.
    """
    with st.container(border=True):
        if title:
            st.markdown(f'<p class="f1-card__title">{escape(title)}</p>', unsafe_allow_html=True)
        yield


def empty_state(title: str, body: str) -> None:
    """Render a "nothing here yet" panel that says what to do about it.

    Args:
        title: What is missing, stated plainly.
        body: The action that would fix it.
    """
    st.markdown(empty_state_html(title, body), unsafe_allow_html=True)
