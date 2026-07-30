"""The design system's markup builders.

These are the pure half of :mod:`f1cp.dashboard.components.theme` -- values in, HTML
string out, no Streamlit involved -- which is what makes them testable at this level at
all. What is worth pinning here is not the exact markup (that is free to change) but the
two properties a caller depends on: the numbers reach the page unaltered, and text from
the API can never break out of the HTML the module builds around it.
"""

from __future__ import annotations

import pytest

from f1cp.dashboard.components import palette
from f1cp.dashboard.components.theme import (
    BoardRow,
    Stat,
    _chip,
    empty_state_html,
    leaderboard_html,
    section_html,
    stat_grid_html,
    stylesheet,
)

pytestmark = pytest.mark.unit


def test_values_are_rendered_exactly_as_given() -> None:
    """The module formats nothing: a caller's string is what the reader sees.

    This is the contract that keeps presentation from quietly disagreeing with the data
    -- rounding belongs to the page that knows what the number means.
    """
    html = stat_grid_html([Stat(label="Title chance", value="50.3%", detail="+9.0% vs Norris")])

    assert "50.3%" in html
    assert "+9.0% vs Norris" in html
    assert "Title chance" in html


def test_accent_tone_marks_exactly_one_card() -> None:
    html = stat_grid_html(
        [
            Stat(label="Favourite", value="Antonelli", tone="accent"),
            Stat(label="Runner-up", value="Norris"),
        ]
    )

    assert html.count("f1-stat--accent") == 1


def test_signed_tones_get_their_own_class() -> None:
    positive = stat_grid_html([Stat(label="Up", value="1", detail="+3", tone="positive")])
    negative = stat_grid_html([Stat(label="Down", value="1", detail="-3", tone="negative")])

    assert "f1-stat__detail--positive" in positive
    assert "f1-stat__detail--negative" in negative


def test_leaderboard_meter_width_tracks_the_share() -> None:
    html = leaderboard_html([BoardRow(position="1", name="Antonelli", value="50.3%", share=0.503)])

    assert "width:50.3%" in html


@pytest.mark.parametrize("share", [-1.0, 1.5])
def test_leaderboard_meter_stays_inside_its_track(share: float) -> None:
    """A share outside 0-1 must not draw a bar wider than the row that contains it."""
    html = leaderboard_html([BoardRow(position="1", name="Driver", value="x", share=share)])

    assert "width:0.0%" in html or "width:100.0%" in html


def test_only_the_leading_row_is_accented() -> None:
    rows = [BoardRow(position=str(i), name=f"Driver {i}", value="1%") for i in range(1, 5)]

    assert leaderboard_html(rows).count("f1-board__row--lead") == 1
    assert leaderboard_html(rows, highlight_first=False).count("f1-board__row--lead") == 0


@pytest.mark.parametrize(
    "build",
    [
        lambda text: _chip(text, variant="accent"),
        lambda text: section_html(text, text),
        lambda text: empty_state_html(text, text),
        lambda text: stat_grid_html([Stat(label=text, value=text, detail=text)]),
        lambda text: leaderboard_html(
            [BoardRow(position=text, name=text, value=text, secondary=text)]
        ),
    ],
)
def test_text_from_the_api_cannot_inject_markup(build) -> None:
    """Driver names, team names and model names all arrive from outside this process.

    Every one of them is interpolated into HTML that Streamlit renders with
    ``unsafe_allow_html=True``, so escaping is the only thing standing between a stray
    angle bracket in the data and a broken -- or hostile -- page.
    """
    html = build("<script>alert('x')</script>")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_stylesheet_is_generated_from_the_palette() -> None:
    """The stylesheet must not carry its own colour literals, or tokens drift from charts."""
    css = stylesheet()

    assert palette.ACCENT in css
    assert palette.SURFACE in css
    assert palette.INK_PRIMARY in css


def test_the_theme_is_emitted_on_every_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: skipping the stylesheet after the first run un-styles the whole app.

    Streamlit rebuilds its element tree from whatever the script emits on each rerun and
    drops anything not re-emitted, so a "inject once per session" guard means the design
    system survives exactly until the user's first click -- after which every page renders
    as unstyled text. This asserts the opposite: repeated calls each emit the stylesheet.
    """
    from f1cp.dashboard.components import theme

    emitted: list[str] = []
    monkeypatch.setattr(theme.st, "html", lambda body, **_: emitted.append(body))

    theme.inject_theme()
    theme.inject_theme()

    assert len(emitted) == 2
    assert all("<style>" in block for block in emitted)


def test_the_stylesheet_goes_through_st_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """`st.markdown` is not the supported path for a `<style>` element.

    ``st.html`` is the API Streamlit documents for raw HTML and CSS, and the one whose
    sanitiser configuration explicitly permits a style tag. Pinning the call keeps the
    stylesheet from drifting back onto the markdown path.
    """
    from f1cp.dashboard.components import theme

    monkeypatch.setattr(
        theme.st, "markdown", lambda *_, **__: pytest.fail("stylesheet must not use st.markdown")
    )
    monkeypatch.setattr(theme.st, "html", lambda body, **_: None)

    theme.inject_theme()
