"""The thinking indicator, and the gate that keeps it off injected streams.

The marker is a background thread writing escape codes, so the interesting
properties are the boundaries: it must never run where there is no terminal to
animate on, and it must never outlive the wait it exists to fill.
"""

from __future__ import annotations

import io
import re
import time

import pytest

from selfagent import render
from selfagent.providers.base import TextDelta
from selfagent.render import ACCENT, Renderer, ThinkingIndicator

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


@pytest.fixture
def vt_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    monkeypatch.setattr(render, "_enable_vt", lambda: True)


# ---- the indicator itself --------------------------------------------


def test_it_animates_in_place_and_erases_itself() -> None:
    out = io.StringIO()
    indicator = ThinkingIndicator(out)
    indicator.start()
    time.sleep(0.25)
    indicator.stop()

    raw = out.getvalue()
    assert "思考中" in raw
    assert raw.count("\r\x1b[K") >= 2, "redrawn in place rather than appended"
    assert raw.rstrip().endswith("\r\x1b[K"), "leaves nothing behind"


def test_it_uses_the_splash_gradient_colour() -> None:
    out = io.StringIO()
    indicator = ThinkingIndicator(out)
    indicator.start()
    time.sleep(0.05)
    indicator.stop()
    r, g, b = ACCENT
    assert f"\x1b[38;2;{r};{g};{b}m" in out.getvalue()


def test_it_shows_elapsed_seconds() -> None:
    out = io.StringIO()
    indicator = ThinkingIndicator(out)
    indicator.start()
    time.sleep(0.15)
    indicator.stop()
    assert re.search(r"\d+\.\d+s", ANSI.sub("", out.getvalue()))


def test_frames_are_ascii() -> None:
    # A braille spinner would be prettier, but U+28xx is absent from a GBK
    # codepage -- and this program has to start on machines that never ran
    # chcp 65001.
    assert render.SPINNER_FRAMES
    assert all(frame.isascii() for frame in render.SPINNER_FRAMES)


def test_stop_before_start_is_safe() -> None:
    ThinkingIndicator(io.StringIO()).stop()


def test_stop_is_idempotent() -> None:
    indicator = ThinkingIndicator(io.StringIO())
    indicator.start()
    indicator.stop()
    indicator.stop()


def test_start_is_idempotent() -> None:
    out = io.StringIO()
    indicator = ThinkingIndicator(out)
    indicator.start()
    indicator.start()
    indicator.stop()
    assert out.getvalue().count("思考中") >= 1


def test_an_unwritable_stream_does_not_raise() -> None:
    class Broken(io.StringIO):
        def write(self, _text: str) -> int:
            raise OSError("closed")

    indicator = ThinkingIndicator(Broken())
    indicator.start()
    time.sleep(0.05)
    indicator.stop()


# ---- the gate --------------------------------------------------------


def test_injected_streams_never_get_a_marker(vt_available: None) -> None:
    # err is not sys.stderr, so there is no cursor to animate on -- and no
    # reason to spawn a thread inside a test.
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    assert renderer._spinner_enabled() is False

    renderer.thinking_start()
    assert renderer._indicator._thread is None


def test_no_terminal_means_no_marker() -> None:
    renderer = Renderer(color=True, out=io.StringIO(), err=io.StringIO())
    assert renderer._spinner_enabled() is False


def test_no_escape_codes_means_no_marker(vt_available: None) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    renderer._vt = False
    assert renderer._spinner_enabled() is False


def test_the_flag_can_turn_it_off() -> None:
    renderer = Renderer(color=True, out=io.StringIO(), err=io.StringIO(), spinner=False)
    assert renderer.show_spinner is False
    assert renderer._spinner_enabled() is False


def test_finish_stops_the_marker(vt_available: None, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    monkeypatch.setattr(renderer, "_spinner_enabled", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(renderer._indicator, "start", lambda: calls.append("start"))
    monkeypatch.setattr(renderer._indicator, "stop", lambda: calls.append("stop"))

    renderer.thinking_start()
    renderer.finish()

    assert calls == ["start", "stop"]


def test_the_first_token_stops_the_marker(vt_available: None, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    monkeypatch.setattr(renderer, "_spinner_enabled", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(renderer._indicator, "start", lambda: calls.append("start"))
    monkeypatch.setattr(renderer._indicator, "stop", lambda: calls.append("stop"))

    renderer.thinking_start()
    renderer.handle(TextDelta("hello"))

    assert calls == ["start", "stop"]


def test_an_empty_delta_leaves_the_marker_running(
    vt_available: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Some endpoints emit empty deltas as keep-alives; they are not output.
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    monkeypatch.setattr(renderer, "_spinner_enabled", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(renderer._indicator, "start", lambda: calls.append("start"))
    monkeypatch.setattr(renderer._indicator, "stop", lambda: calls.append("stop"))

    renderer.thinking_start()
    renderer.handle(TextDelta(""))

    assert calls == ["start"]
