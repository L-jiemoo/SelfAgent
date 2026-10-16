"""The animated startup logo.

The animation is pure presentation, so what these tests pin down is not that it
looks good but that it never leaks: no escape codes when stdout is not a
console, no *cursor* codes on a console that will not interpret them, and a
final frame identical to the static splash.
"""

from __future__ import annotations

import io
import re

import pytest

from selfagent import banner, render
from selfagent.render import Renderer

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CURSOR_UP = re.compile(r"\x1b\[\d+F")


class Tty(io.StringIO):
    """Reports a real terminal. `encoding` must be a class attribute -- io
    .StringIO's own is a read-only descriptor."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def plain(text: str) -> str:
    return ANSI.sub("", text)


def logo_rows() -> list[str]:
    return [line.rstrip() for line in banner.LOGO]


@pytest.fixture
def vt_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    monkeypatch.setattr(render, "_enable_vt", lambda: True)


@pytest.fixture
def vt_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    monkeypatch.setattr(render, "_enable_vt", lambda: False)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # The animation is defined by its frames, not by how long it lingers.
    monkeypatch.setattr(render.time, "sleep", lambda _seconds: None)


def draw(renderer: Renderer, info=None) -> str:
    renderer.splash(
        banner.LOGO,
        info if info is not None else [("标题", "bold")],
        top=banner.COLOR_TOP,
        bottom=banner.COLOR_BOTTOM,
    )
    return renderer.out.getvalue()


# ---- gating ----------------------------------------------------------


def test_injected_stream_is_never_animated(vt_available: None) -> None:
    # Tests and embedders hand us their own stream; animating into it would
    # mean sleeping through frames nobody can see.
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    assert renderer.animate is False


def test_animation_is_off_without_a_terminal() -> None:
    renderer = Renderer(color=True, out=io.StringIO(), err=io.StringIO())
    assert renderer.animate is False


def test_animation_is_off_without_colour(vt_available: None) -> None:
    renderer = Renderer(color=False, out=Tty(), err=io.StringIO())
    assert renderer.animate is False


def test_animation_can_be_switched_off(vt_available: None, no_sleep: None) -> None:
    text = draw(Renderer(color=True, out=Tty(), err=io.StringIO(), animate=False))
    assert not CURSOR_UP.search(text)
    assert "38;2;34;211;238" in text  # the still gradient survives


# ---- the animated path -----------------------------------------------


def test_animated_splash_moves_the_cursor(vt_available: None, no_sleep: None) -> None:
    text = draw(Renderer(color=True, out=Tty(), err=io.StringIO(), animate=True))
    assert CURSOR_UP.search(text), "the logo is redrawn, so it must reposition"
    assert "\x1b[2K" in text, "rows are erased before being rewritten"
    assert "\x1b[?25l" in text and "\x1b[?25h" in text, "cursor is hidden then restored"
    assert "标题" in text, "the info lines still follow the logo"


def test_animated_splash_settles_on_the_static_frame(
    vt_available: None, no_sleep: None
) -> None:
    text = draw(Renderer(color=True, out=Tty(), err=io.StringIO(), animate=True))
    body = plain(text)
    for row in logo_rows():
        assert row in body
    assert "38;2;34;211;238" in text  # top of the gradient
    assert "38;2;168;85;247" in text  # bottom of the gradient


def test_animated_splash_ends_without_a_stray_newline_gap(
    vt_available: None, no_sleep: None
) -> None:
    # The cursor is left directly under the last logo row, so the info block
    # begins on the very next line rather than after a run of blank ones.
    text = draw(Renderer(color=True, out=Tty(), err=io.StringIO(), animate=True))
    assert "\n\n\n" not in plain(text)


# ---- degradation -----------------------------------------------------


def test_reveal_fallback_when_vt_cannot_be_enabled(
    vt_unavailable: None, no_sleep: None
) -> None:
    text = draw(Renderer(color=True, out=Tty(), err=io.StringIO(), animate=True))
    assert not CURSOR_UP.search(text), "no cursor control on a console that lacks it"
    assert "\x1b[2K" not in text
    body = plain(text)
    for row in logo_rows():
        assert row in body


# ---- frame helpers ---------------------------------------------------


def test_blend_endpoints_and_clamping() -> None:
    assert render._blend((1, 2, 3), (9, 9, 9), 0.0) == (1, 2, 3)
    assert render._blend((1, 2, 3), (9, 9, 9), 1.0) == (9, 9, 9)
    assert render._blend((1, 2, 3), (9, 9, 9), 5.0) == (9, 9, 9)
    assert render._blend((1, 2, 3), (9, 9, 9), -1.0) == (1, 2, 3)


def test_sweep_row_preserves_the_text() -> None:
    row = render._sweep_row("a b", (10, 20, 30), 0.0, 3)
    assert plain(row) == "a b"


def test_plain_row_preserves_the_text() -> None:
    assert plain(render._plain_row("a b", (10, 20, 30))) == "a b"


def test_glint_peaks_under_its_phase() -> None:
    base = (10, 20, 30)
    assert render._sweep_row("ab", base, 0.0, 2).startswith(render._fg(render.HIGHLIGHT))


def test_glint_wraps_around_the_loop() -> None:
    # A glint at phase 0.98 is still lighting the left edge, because distance is
    # measured around the infinity rather than along the row.
    base = (10, 20, 30)
    assert not render._sweep_row("ab", base, 0.98, 2).startswith(render._fg(base))


def test_row_far_from_the_glint_keeps_the_base_colour() -> None:
    base = (10, 20, 30)
    row = render._sweep_row("abcd", base, 0.0, 4)
    assert render._fg(base) in row
