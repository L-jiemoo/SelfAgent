"""The gradient box drawn around the REPL's input line.

It is drawn with escape codes, so the interesting properties are again the
boundaries: it must not appear where escape codes are not interpreted, it must
always be closed, and it must fall back to glyphs a GBK console can print.
"""

from __future__ import annotations

import builtins
import io
import re
import sys

import pytest

from selfagent import render
from selfagent.render import (
    ACCENT,
    BOX_COMMANDS,
    BOX_H,
    BOX_H_FALLBACK,
    BOX_MARKER,
    BOX_MARKER_FALLBACK,
    GRADIENT_BOTTOM,
    GRADIENT_TOP,
    Renderer,
    _gradient_rule,
    _hint_fragments,
    _input_box_width,
    _input_text_width,
    _reflow_columns,
    _rewrapped_rows_above_cursor,
    _row_content_width,
    _rule_tokens,
    _terminal_columns,
)

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

ACCENT_ESCAPE = "\x1b[38;2;34;211;238m"


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class GbkTty(Tty):
    encoding = "gbk"


class Char:
    def __init__(self, char: str, width: int = 1) -> None:
        self.char = char
        self.width = width


def row(text: str, padded_to: int | None = None) -> dict[int, Char]:
    cells = {index: Char(char) for index, char in enumerate(text)}
    if padded_to is not None:
        for index in range(len(text), padded_to):
            cells[index] = Char(" ")
    return cells


@pytest.fixture
def vt_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    monkeypatch.setattr(render, "_enable_vt", lambda: True)


@pytest.fixture
def boxed(monkeypatch: pytest.MonkeyPatch, vt_available: None) -> Renderer:
    """A renderer that will actually draw the box, over an inspectable stream."""
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    monkeypatch.setattr(renderer, "_box_enabled", lambda: True)
    return renderer


def answer(monkeypatch: pytest.MonkeyPatch, text: str) -> list[str]:
    """Replace input(), recording the prompt it was handed."""
    seen: list[str] = []

    def fake(prompt: str = "") -> str:
        seen.append(prompt)
        return text

    monkeypatch.setattr(builtins, "input", fake)
    return seen


# ---- the rule --------------------------------------------------------


def test_the_rule_spans_the_requested_width() -> None:
    rule = _gradient_rule(60, BOX_H, GRADIENT_TOP, GRADIENT_BOTTOM)
    assert len(ANSI.sub("", rule)) == 60


def test_the_rule_runs_from_the_cyan_end_to_the_violet_end() -> None:
    rule = _gradient_rule(40, BOX_H, GRADIENT_TOP, GRADIENT_BOTTOM)
    assert f"\x1b[38;2;{GRADIENT_TOP[0]};{GRADIENT_TOP[1]};{GRADIENT_TOP[2]}m" in rule
    assert f"\x1b[38;2;{GRADIENT_BOTTOM[0]};{GRADIENT_BOTTOM[1]};{GRADIENT_BOTTOM[2]}m" in rule


def test_the_rule_uses_the_splash_gradient() -> None:
    # Same numbers as the logo and the spinner, so the three look related.
    assert GRADIENT_TOP == ACCENT


# ---- glyph selection -------------------------------------------------


def test_unicode_glyphs_are_used_on_a_utf8_console(vt_available: None) -> None:
    chars = Renderer(color=True, out=Tty(), err=io.StringIO())._box_chars()
    assert chars["h"] == BOX_H
    assert chars["marker"] == BOX_MARKER


def test_ascii_glyphs_when_the_rule_cannot_be_encoded() -> None:
    class Narrow(Tty):
        encoding = "ascii"

    chars = Renderer(color=True, out=Narrow(), err=io.StringIO())._box_chars()
    assert chars["h"] == BOX_H_FALLBACK
    assert chars["marker"] == BOX_MARKER_FALLBACK


def test_the_marker_falls_back_on_a_gbk_console(vt_available: None) -> None:
    # U+276F is not in GBK, so the Claude-style arrow becomes a plain '>'.
    chars = Renderer(color=True, out=GbkTty(), err=io.StringIO())._box_chars()
    assert chars["marker"] == BOX_MARKER_FALLBACK
    assert chars["h"] == BOX_H, "the rule itself is still GBK-safe"


def test_box_width_leaves_a_column_of_slack(vt_available: None) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    assert renderer._box_width() <= 100


# ---- the gate --------------------------------------------------------


def test_injected_streams_get_the_plain_prompt(vt_available: None, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    assert renderer._box_enabled() is False

    seen = answer(monkeypatch, "hi")
    assert renderer.prompt() == "hi"
    assert seen == ["\n你> "]


def test_no_escape_codes_means_no_box(vt_available: None, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    renderer._vt = False
    assert renderer._box_enabled() is False

    answer(monkeypatch, "hi")
    renderer.prompt()
    assert BOX_H not in renderer.out.getvalue()


def test_the_flag_turns_the_box_off(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=io.StringIO(), err=io.StringIO(), prompt_box=False)
    assert renderer.show_box is False
    assert renderer._box_enabled() is False


# ---- drawing ---------------------------------------------------------


def test_two_rules_sandwich_the_prompt(boxed: Renderer, monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, "你好")
    assert boxed.prompt() == "你好"

    rows = ANSI.sub("", boxed.out.getvalue()).split("\n")
    rules = [row for row in rows if row and set(row) == {BOX_H}]
    assert len(rules) == 2, "one rule above and one below"
    assert f"{BOX_MARKER} " in "\n".join(rows), "the marker opens the input line"
    assert "│" not in "\n".join(rows), "no side walls"


def test_the_prompt_keeps_the_gradient(boxed: Renderer, monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, "hi")
    boxed.prompt()
    raw = boxed.out.getvalue()
    assert f"\x1b[38;2;{GRADIENT_TOP[0]};{GRADIENT_TOP[1]};{GRADIENT_TOP[2]}m" in raw
    assert f"\x1b[38;2;{GRADIENT_BOTTOM[0]};{GRADIENT_BOTTOM[1]};{GRADIENT_BOTTOM[2]}m" in raw


def test_terminal_width_is_read_live_not_cached() -> None:
    # Resizing the window while the prompt is open must move the box's clamp:
    # a width captured at session start would leave the rule off-screen after
    # the window shrinks.
    class Out:
        columns = 120

        def get_size(self):
            return type("Size", (), {"columns": self.columns, "rows": 40})()

    out = Out()
    assert _terminal_columns(out) == 120
    out.columns = 60
    assert _terminal_columns(out) == 60
    assert _input_box_width(500, _terminal_columns(out)) == 58


# ---- reflow after a resize ------------------------------------------
# Windows Terminal rewraps the old frame before prompt_toolkit hears about the
# resize. The extra rule rows it creates have to be erased too, or they stay
# behind as ghost horizontal lines.


def test_row_content_width_ignores_the_padded_blanks() -> None:
    assert _row_content_width(row("────────", padded_to=80)) == 8


def test_a_wrapped_rule_is_counted_above_the_cursor() -> None:
    # 8 columns of rule at 6 columns per row occupy two physical rows.
    assert _rewrapped_rows_above_cursor([row("────────")], 2, 1, 6) == 2


def test_a_rule_that_still_fits_is_counted_once() -> None:
    assert _rewrapped_rows_above_cursor([row("────────")], 2, 1, 20) == 1


def test_wrapped_segments_of_the_cursor_line_are_counted() -> None:
    # One row above the cursor, plus the wrapped part of the cursor's own line.
    assert _rewrapped_rows_above_cursor([row("")], 10, 1, 6) == 2


def test_win32_width_includes_the_reserved_last_column() -> None:
    class Size:
        columns = 59

    class Win32Like:
        def get_win32_screen_buffer_info(self) -> None: ...

    class Plain:
        pass

    assert _reflow_columns(Win32Like(), Size()) == 60
    assert _reflow_columns(Plain(), Size()) == 59


# ---- measuring the rewrapped screen ----------------------------------
# The console buffer is reflowed along with the display, so it can be asked
# where the frame actually ended up instead of predicting the rewrap.


class FakeScreen:
    def __init__(self, rows: list[str]) -> None:
        self.data_buffer = [row(text) for text in rows]
        self.height = len(rows)


class FakeInfo:
    def __init__(self, caret_x: int, caret_y: int, columns: int) -> None:
        self.dwCursorPosition = type("Pos", (), {"X": caret_x, "Y": caret_y})()
        self.srWindow = type(
            "Rect", (), {"Left": 0, "Top": 0, "Right": columns - 1, "Bottom": 20}
        )()


class FakeConsoleOutput:
    def __init__(self, caret: tuple[int, int], columns: int) -> None:
        self.caret = caret
        self.columns = columns

    def get_win32_screen_buffer_info(self) -> FakeInfo:
        return FakeInfo(self.caret[0], self.caret[1], self.columns)


def measure(
    monkeypatch: pytest.MonkeyPatch,
    screen_rows: list[str],
    buffer_rows: dict[int, str],
    caret: tuple[int, int],
    columns: int,
    old_columns: int = 80,
    line_count: int = 1,
) -> tuple[int, int] | None:
    monkeypatch.setattr(render, "_read_console_row", lambda row, width: buffer_rows.get(row))
    return render._measured_erase_target(
        FakeConsoleOutput(caret, columns),
        FakeScreen(screen_rows),
        caret[0],
        1,
        old_columns,
        BOX_MARKER,
        BOX_H,
        line_count,
    )


def test_measured_target_uses_the_unwrapped_frame_when_nothing_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = measure(
        monkeypatch,
        ["─" * 8, "❯ hello"],
        {0: "─" * 8, 1: "❯ hello"},
        caret=(4, 1),
        columns=40,
    )
    assert target == (4, 1)


def test_measured_target_counts_the_rewrapped_top_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 8-column rule was rewrapped into two rows at 6 columns; the caret is
    # now on row 2, so both rule rows have to be erased.
    target = measure(
        monkeypatch,
        ["─" * 8, "❯ hi"],
        {0: "─" * 6, 1: "─" * 2, 2: "❯ hi"},
        caret=(3, 2),
        columns=6,
        old_columns=8,
    )
    assert target == (3, 2)


def test_measured_target_finds_the_marker_when_the_caret_line_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The input row itself was split: the caret sits on the continuation row,
    # which does not start with the marker. Walk up to the marker row.
    target = measure(
        monkeypatch,
        ["─" * 8, "❯ abcdefgh"],
        {0: "─" * 6, 1: "❯ abcdef", 2: "gh"},
        caret=(2, 2),
        columns=6,
        old_columns=8,
    )
    assert target == (2, 2)


def test_measured_target_stops_at_the_old_top_rule_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A rule-looking conversation line directly above the frame must not be
    # counted: the old top rule was six columns and at six columns is one row.
    target = measure(
        monkeypatch,
        ["─" * 6, "❯ hi"],
        {0: "─" * 6, 1: "─" * 6, 2: "❯ hi"},
        caret=(3, 2),
        columns=6,
        old_columns=6,
    )
    assert target == (3, 1)


def test_measured_target_also_works_when_the_window_grew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Full repaint is used for both directions; growing cannot rewrap a row,
    # so the measured frame is the plain logical one.
    target = measure(
        monkeypatch,
        ["─" * 20, "❯ hello"],
        {0: "─" * 20, 1: "❯ hello"},
        caret=(7, 1),
        columns=100,
        old_columns=40,
    )
    assert target == (7, 1)


def test_measured_target_finds_the_marker_above_a_continuation_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pasted block puts the caret on a later input line, which begins with
    # the indent rather than the marker.
    target = measure(
        monkeypatch,
        ["─" * 10, "❯ abc", "  def"],
        {0: "─" * 10, 1: "❯ abc", 2: "  def"},
        caret=(5, 2),
        columns=40,
        old_columns=40,
        line_count=2,
    )
    assert target == (5, 2)


def test_measured_target_gives_up_when_the_marker_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whatever is on screen is not our frame -- let the caller fall back
    # rather than erase an unknown number of rows.
    assert (
        measure(
            monkeypatch,
            ["─" * 8, "❯ hello"],
            {0: "──────", 1: "ordinary output"},
            caret=(4, 1),
            columns=40,
        )
        is None
    )


def test_rule_tokens_carry_the_configured_end_colours() -> None:
    tokens = _rule_tokens(8, BOX_H, (255, 0, 0), (0, 0, 255))
    assert len(tokens) == 8
    assert tokens[0] == ("#ff0000", BOX_H)
    assert tokens[-1] == ("#0000ff", BOX_H)


def test_the_prompt_follows_the_configured_theme(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    boxed.set_theme((255, 0, 0), (0, 0, 255), (0, 255, 0))
    answer(monkeypatch, "hi")
    boxed.prompt()

    raw = boxed.out.getvalue()
    assert "\x1b[38;2;255;0;0m" in raw, "top rule uses the configured top"
    assert "\x1b[38;2;0;0;255m" in raw, "bottom rule and end wall use the bottom"
    assert "\x1b[38;2;0;255;0m" in raw, "the marker uses the accent"


def test_the_cursor_steps_past_the_box_even_when_input_fails(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", explode)

    with pytest.raises(EOFError):
        boxed.prompt()

    # The frame is already on screen, so the only thing left to do is leave it --
    # otherwise the REPL's parting newline would land inside the box.
    assert ANSI.sub("", boxed.out.getvalue()).endswith("\n\n")


def test_the_plain_prompt_is_not_closed_with_a_box(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Renderer(color=True, out=io.StringIO(), err=io.StringIO())
    answer(monkeypatch, "hi")
    renderer.prompt()
    assert BOX_H not in renderer.out.getvalue()


# ---- width tracking --------------------------------------------------
# The prompt_toolkit box is redrawn on every keystroke, so its WIDTH must track
# the typed length: grow as characters are added, shrink as they are deleted,
# and never exceed the terminal or collapse below a readable sliver.


def test_the_empty_box_is_a_readable_sliver() -> None:
    # 8 = the minimum (chrome + a little room), so an empty prompt is still a box.
    assert _input_box_width(0, 120) == 8


def test_the_box_grows_with_the_text() -> None:
    # +2 for the "❯ " marker.
    assert _input_box_width(10, 120) == 12
    assert _input_box_width(20, 120) == 22
    # A line shorter than the minimum floor is clamped, not pinned wide.
    assert _input_box_width(3, 120) == 8


def test_the_box_uses_wide_characters_for_width() -> None:
    # "你好世界" is 8 columns, so the rule should be 8 + 2 = 10 columns.
    # The helper takes a precomputed column count, so 8 here means 8 columns.
    assert _input_box_width(8, 120) == 10


def test_the_box_stops_at_the_terminal_width() -> None:
    # term_w - BOX_MARGIN (2) is the ceiling; text longer than that scrolls.
    assert _input_box_width(500, 100) == 98


def test_the_text_column_reserves_the_caret_cell() -> None:
    # The box is sized from the longest line plus one cell the caret occupies,
    # so 19 columns of text leave a 20-column typing column and a 22-wide rule.
    assert _input_box_width(19 + 1, 120) == 22
    assert _input_text_width(_input_box_width(19 + 1, 120)) == 20
    # At the terminal clamp the column is whatever the rule leaves; the buffer
    # scrolls inside it rather than pushing the rules outwards.
    box = _input_box_width(500, 120)
    assert _input_text_width(box) == box - 2


def hint_text(line: str, columns: int = 120, accent: str = "#22d3ee") -> str:
    return "".join(text for _style, text in _hint_fragments(line, columns, accent))


def test_the_empty_hint_advertises_shortcuts_and_commands() -> None:
    hint = hint_text("")
    assert hint.startswith("  /")
    assert "Enter 发送" in hint and "Ctrl+C 退出" in hint
    assert render._visible_width(hint) <= 120


def test_typing_a_slash_lists_commands() -> None:
    hint = hint_text("/")
    assert "/help" in hint and "/model" in hint
    assert "/theme" in hint and "/name" in hint


def test_command_suggestions_are_whole_entries() -> None:
    # At 80 columns /status no longer fits after /name. Dropping it is better
    # than showing half a command name that cannot be typed from.
    hint = hint_text("/", 80)
    assert "/name" in hint
    assert "/sta" not in hint
    assert "当前状态" not in hint


def test_typing_a_prefix_narrows_the_list_live() -> None:
    # Each extra character is one more keystroke; the hint follows the buffer,
    # so by the time '/mo' is on screen only the command it can become is left.
    assert "/model" in hint_text("/m")
    assert "/model" in hint_text("/mo")
    assert "/model" in hint_text("/mod")
    assert "/theme" not in hint_text("/mod")
    assert "/help" not in hint_text("/mod")


def test_a_complete_command_explains_itself() -> None:
    hint = hint_text("/theme #22d3ee #a855f7")
    assert "/theme" in hint
    assert "改配色" in hint


def test_an_unknown_command_says_so() -> None:
    hint = hint_text("/nonsense")
    assert "无匹配" in hint and "/help" in hint


def test_plain_text_gets_the_submit_shortcut() -> None:
    hint = hint_text("你好世界")
    assert "Enter 发送" in hint
    assert "/" not in hint


def test_the_hint_highlights_the_typed_command() -> None:
    fragments = _hint_fragments("/mo", 120, "#ff0000")
    assert ("#ff0000", "/model") in fragments


def test_the_hint_is_cut_rather_than_wrapped_on_a_narrow_terminal() -> None:
    hint = hint_text("/", 12)
    assert render._visible_width(hint) <= 12
    assert hint.startswith("  ")
    assert "\n" not in hint


def test_the_hint_mentions_the_customisation_commands() -> None:
    names = {name for name, _description in BOX_COMMANDS}
    assert {"/theme", "/name"} <= names
    # Both command spellings are listed, as in /help.
    assert {"/usage", "/cost", "/exit", "/quit"} <= names


def test_rows_below_cursor_tolerates_outputs_that_cannot_answer() -> None:
    class Silent:
        def get_rows_below_cursor_position(self) -> int:
            raise NotImplementedError

    class Answering:
        def get_rows_below_cursor_position(self) -> int:
            return 7

    assert render._rows_below_cursor(Silent()) is None
    assert render._rows_below_cursor(Answering()) == 7


def test_docking_extra_counts_the_rows_after_the_caret() -> None:
    # Single line, 4-row frame: the caret ends 3 rows above the bottom.
    assert render._docking_extra(10, 4, 0) == 7
    assert render._docking_extra(3, 4, 0) == 0
    # A three-line input has a 6-row frame, but with the caret on its last
    # line it also wants to end 3 rows above the bottom -- not 5.
    assert render._docking_extra(8, 6, 2) == 5
    assert render._docking_extra(None, 6, 2) == 0


def test_bottom_plan_slides_the_window_when_scrollback_exists() -> None:
    # Long session: the frame's last row is buffer row 502, the window ends at
    # 509. Moving the window up exposes seven more rows of history above.
    assert render._bottom_plan(500, 480, 509, 9001, 3) == ("move", 473)
    # Already docked: nothing to do.
    assert render._bottom_plan(505, 480, 507, 9001, 3) == ("none", 0)
    # A shrink clipped two rows: slide the window down so they are visible.
    assert render._bottom_plan(505, 480, 499, 9001, 3) == ("move", 488)


def test_bottom_plan_shifts_the_block_when_the_window_cannot_move() -> None:
    # Fresh tab: frame's last row is 11 in a 26-row buffer, so no window
    # position can put it at row 25. The block itself moves down 14 rows.
    assert render._bottom_plan(9, 0, 25, 26, 3) == ("shift", 14)
    # Frame already on the bottom row.
    assert render._bottom_plan(23, 0, 25, 26, 3) == ("none", 0)
    # A shrink discarded the rows below the caret; the frame must be redrawn.
    assert render._bottom_plan(18, 0, 19, 20, 3) == ("repaint", 0)


def test_moving_to_the_bottom_steps_down_by_the_blank_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = Renderer(color=True, out=sys.stdout, err=io.StringIO())
    monkeypatch.setattr(renderer, "_box_enabled", lambda: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    import prompt_toolkit.application.application as appmod
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    captured: dict[str, object] = {}
    original_init = appmod.Application.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("output", DummyOutput())
        kwargs.setdefault("input", DummyInput())
        original_init(self, *args, **kwargs)
        captured["app"] = self

    monkeypatch.setattr(appmod.Application, "__init__", patched_init)
    monkeypatch.setattr(appmod.Application, "run", lambda self, **kwargs: "typed text")
    assert renderer._prompt_toolkit("你") == "typed text"

    app = captured["app"]
    from prompt_toolkit.layout import Window
    from prompt_toolkit.layout.controls import BufferControl

    def find_buffer(container):
        if isinstance(container, Window) and isinstance(container.content, BufferControl):
            return container.content.buffer
        for child in container.get_children():
            found = find_buffer(child)
            if found is not None:
                return found
        return None

    # preferred_height() asks the buffer to load history, which needs a running
    # event loop. There is none in a unit test; mark it done.
    find_buffer(app.layout.container)._load_history_task = object()

    calls: list[object] = []
    monkeypatch.setattr(app.output, "get_size", lambda: Size(rows=30, columns=80))
    monkeypatch.setattr(app.output, "get_rows_below_cursor_position", lambda: 25)
    monkeypatch.setattr(app.output, "cursor_down", lambda n: calls.append(n))
    monkeypatch.setattr(app.output, "flush", lambda: None)
    monkeypatch.setattr(
        app.renderer,
        "erase",
        lambda leave_alternate_screen=True: calls.append("erase"),
    )
    monkeypatch.setattr(
        app, "_request_absolute_cursor_position", lambda: calls.append("cpr")
    )
    monkeypatch.setattr(app, "_redraw", lambda: calls.append("redraw"))

    app._move_to_bottom(4, 0)

    # The frame is 4 rows (top rule, input, bottom rule, live hint), so the
    # caret stops 3 rows above the bottom: 25 - 3 = 22 rows down.
    assert calls == ["erase", 22, "cpr", "redraw"]


def test_the_rule_styles_are_valid_prompt_toolkit_colours() -> None:
    # A stray '#' turns a colour into '##rrggbb'; prompt_toolkit raises on the
    # first repaint and the whole takeover silently falls back to the static
    # input() box. Every fragment built here must parse.
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.styles import Style

    parser = Style([])
    for width in (8, 20):
        for style, _text in _rule_tokens(width, BOX_H):
            assert style.startswith("#")
            assert not style.startswith("##")
            parser.get_attrs_for_style_str(style)  # raises ValueError if malformed


# ---- prompt_toolkit takeover is gated on a real console -----------------


def test_the_terminal_takeover_yields_to_the_fallback_when_out_is_not_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The prompt_toolkit path only runs against the real sys.stdout. Any other
    # stream -- a test's StringIO, a redirect, an embedder -- must get the
    # static input() box so the existing fallback contract holds.
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    monkeypatch.setattr(renderer, "_box_enabled", lambda: True)
    assert renderer._prompt_toolkit("你") is None


def test_the_terminal_takeover_is_skipped_when_prompt_toolkit_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the bundle somehow omitted prompt_toolkit, the box must degrade rather
    # than crash at import time.
    renderer = Renderer(color=True, out=sys.stdout, err=io.StringIO())
    monkeypatch.setattr(renderer, "_box_enabled", lambda: True)

    real_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "prompt_toolkit" or name.startswith("prompt_toolkit."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    assert renderer._prompt_toolkit("你") is None


def test_the_takeover_path_builds_and_returns_the_entered_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The width-tracking Application is built and its run() reached. We stub the
    # terminal check and inject console-free I/O so it constructs headlessly (a
    # real console is only needed for the actual event loop), then stub run() so
    # the happy path yields the entered text -- proving the layout builds and the
    # result is returned.
    renderer = Renderer(color=True, out=sys.stdout, err=io.StringIO())
    monkeypatch.setattr(renderer, "_box_enabled", lambda: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    import prompt_toolkit.application.application as appmod
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    original_init = appmod.Application.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("output", DummyOutput())
        kwargs.setdefault("input", DummyInput())
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(appmod.Application, "__init__", patched_init)
    monkeypatch.setattr(appmod.Application, "run", lambda self, **kwargs: "typed text")
    assert renderer._prompt_toolkit("你") == "typed text"


# ---- the frame is complete while typing ------------------------------


def test_visible_width_counts_cjk_as_two_columns() -> None:
    assert render._visible_width("ab") == 2
    assert render._visible_width("你好") == 4


def test_visible_width_ignores_escape_codes() -> None:
    assert render._visible_width(f"{ACCENT_ESCAPE}ab\x1b[0m") == 2


def test_visible_width_treats_the_marker_as_one_column() -> None:
    # U+276F is East Asian Neutral, so it really is one column -- if it were
    # wide the cursor would be put in the wrong place.
    assert render._visible_width(BOX_MARKER) == 1


def test_the_whole_frame_exists_before_a_single_key_is_read(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake(prompt: str = "") -> str:
        seen.append(ANSI.sub("", boxed.out.getvalue()))
        return "hi"

    monkeypatch.setattr(builtins, "input", fake)
    boxed.prompt()

    while_typing = seen[0]
    rules = [line for line in while_typing.split("\n") if line and set(line) == {BOX_H}]
    assert len(rules) == 2, "both rules already drawn"


def test_the_input_row_has_no_side_walls(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake(prompt: str = "") -> str:
        seen.append(ANSI.sub("", boxed.out.getvalue()))
        return "hi"

    monkeypatch.setattr(builtins, "input", fake)
    boxed.prompt()

    rows = [line for line in seen[0].split("\n") if BOX_MARKER in line]
    assert len(rows) == 1
    assert rows[0].startswith(f"{BOX_MARKER} ")
    assert "│" not in rows[0]


def test_every_row_of_the_box_is_the_same_width(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "hi")
    boxed.prompt()

    rows = [line for line in ANSI.sub("", boxed.out.getvalue()).split("\n") if line.strip()]
    assert {render._visible_width(row) for row in rows} == {boxed._box_width()}


def test_a_chinese_answer_does_not_shift_the_frame(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The width maths is about the frame, not the text, so the text cannot
    # affect it -- but wide characters are exactly where that assumption breaks.
    answer(monkeypatch, "你好世界")
    boxed.prompt()

    rows = [line for line in ANSI.sub("", boxed.out.getvalue()).split("\n") if line.strip()]
    assert {render._visible_width(row) for row in rows} == {boxed._box_width()}


def test_the_cursor_is_moved_back_inside_the_frame(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = answer(monkeypatch, "hi")
    boxed.prompt()

    assert seen == [""], "the prompt is drawn by us, so input() gets nothing to print"
    raw = boxed.out.getvalue()
    assert "\x1b[2A" in raw, "two rows back up onto the input row"
    assert "\x1b[3G" in raw, "and across to just past the marker"


def test_the_answer_starts_below_the_box(boxed: Renderer, monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, "hi")
    boxed.prompt()
    # The bottom rule is already on screen, so the cursor only has to step past
    # it -- the caller's next write lands below the frame.
    assert ANSI.sub("", boxed.out.getvalue()).endswith("\n\n")


# ---- long lines are compressed ---------------------------------------


def test_compression_keeps_both_ends() -> None:
    text = "A" * 100 + "MIDDLE" + "B" * 100

    shown = render._compress_for_box(text, 40)

    assert render._visible_width(shown) <= 40
    assert shown.startswith("A")
    assert shown.endswith("B")
    assert "MIDDLE" not in shown, "the middle is the part that can be spared"


def test_compression_reports_the_full_length() -> None:
    # So the user can tell their whole paste arrived even though only part of it
    # is on screen.
    assert "500" in render._compress_for_box("x" * 500, 60)


def test_a_short_line_is_left_alone() -> None:
    assert render._compress_for_box("hello", 40) == "hello"


def test_compression_counts_wide_characters() -> None:
    shown = render._compress_for_box("字" * 100, 40)
    assert render._visible_width(shown) <= 40


def test_a_tiny_allowance_does_not_crash() -> None:
    assert render._compress_for_box("hello world", 0) == ""
    assert render._compress_for_box("hello world", 1)


def test_a_short_line_is_not_repainted(boxed: Renderer, monkeypatch: pytest.MonkeyPatch) -> None:
    answer(monkeypatch, "hi")
    boxed.prompt()
    assert "\x1b[1A" not in boxed.out.getvalue()


def test_a_long_line_is_compressed_in_the_box(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "x" * 500)

    boxed.prompt()

    raw = boxed.out.getvalue()
    assert "\x1b[0K" in raw, "the text that spilled out is erased"
    assert "共500字" in ANSI.sub("", raw)


def test_a_long_line_is_still_submitted_in_full(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "y" * 500
    answer(monkeypatch, text)
    assert boxed.prompt() == text, "only the display is shortened"


def test_the_repaint_walks_back_the_rows_the_input_used(
    boxed: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "x" * 500)
    boxed.prompt()

    rows = boxed._input_rows(4, 500)
    assert rows > 1
    assert f"\x1b[{rows}A" in boxed.out.getvalue()
