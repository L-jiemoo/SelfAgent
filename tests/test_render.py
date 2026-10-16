"""The renderer's two output paths.

Every method that prints has a rich branch (colour on) and a plain branch
(colour off). The rest of the suite only ever builds a colourless Renderer, so
the rich branch -- where rich's own API rules apply, notably that
Console.print() takes no file= -- went completely unexercised until it crashed
on a real console. These tests build a colour Renderer over StringIO and call
each method, asserting both that it does not raise and that stderr messages
stay off stdout.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from selfagent import render
from selfagent.providers.base import ToolCall, ToolResult
from selfagent.render import Renderer


@pytest.fixture
def color_renderer(monkeypatch: pytest.MonkeyPatch) -> Renderer:
    # use_color is gated on isatty, which StringIO always fails.
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    return Renderer(color=True, out=io.StringIO(), err=io.StringIO())


def test_color_mode_builds_both_consoles(color_renderer: Renderer) -> None:
    assert color_renderer._console is not None
    assert color_renderer._err_console is not None


def test_stderr_messages_stay_off_stdout(color_renderer: Renderer) -> None:
    color_renderer.warning("注意")
    color_renderer.error("炸了")
    color_renderer.tool_start(
        ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"})
    )
    color_renderer.tool_end(ToolResult(id="c1", content="失败:没了", is_error=True))

    err = color_renderer.err.getvalue()
    assert "注意" in err
    assert "炸了" in err
    assert "read_file" in err
    assert "没了" in err
    assert color_renderer.out.getvalue() == ""


def test_diff_goes_to_stderr(color_renderer: Renderer) -> None:
    color_renderer.diff("写入", Path("a.txt"), "old\n", "new\n")

    err = color_renderer.err.getvalue()
    assert "模型请求写入" in err
    assert "+new" in err
    assert "-old" in err
    assert color_renderer.out.getvalue() == ""


def test_splash_and_status_go_to_stdout(color_renderer: Renderer) -> None:
    color_renderer.splash(("LOGO",), [("标题", "bold")])
    color_renderer.status("状态")

    out = color_renderer.out.getvalue()
    assert "LOGO" in out
    assert "标题" in out
    assert color_renderer.err.getvalue() == ""


def test_plain_mode_still_writes_the_same_text() -> None:
    out, err = io.StringIO(), io.StringIO()
    renderer = Renderer(color=False, out=out, err=err)

    renderer.warning("注意")
    renderer.diff("写入", Path("a.txt"), "old\n", "new\n")

    assert "注意" in err.getvalue()
    assert "模型请求写入" in err.getvalue()
    assert out.getvalue() == ""


def test_draining_pending_keys_is_safe_without_a_console() -> None:
    # Runs for real on Windows (msvcrt) and must be a no-op elsewhere.
    render._drain_pending_keys()
