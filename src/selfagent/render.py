"""Streaming output that adapts to where it is going.

Two very different consumers read this program's stdout. A human at a terminal
wants colour and a dimmed thinking trace. A shell redirect wants clean text with
no escape codes at all -- `agent "q" > out.txt` must produce a file you can paste
straight into another tool. So colour is gated on isatty, and thinking output is
suppressed entirely when not on a terminal.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, TextIO

from .providers.base import (
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    Usage,
)

MAX_DIFF_LINES = 200

# ---- splash animation -------------------------------------------------
# The logo itself is a still gradient; the animation is a glint that runs
# along it. Timings are deliberately short: this program is expected to start
# from a USB stick, and a splash that costs a full second is a splash people
# learn to resent. `ui.splash_animation` in config.json turns it off entirely.
HIGHLIGHT = (0xE8, 0xFB, 0xFF)  # icy glint, reads as light at either gradient end
REVEAL_STEP = 0.045             # seconds between rows appearing
SWEEP_STEPS = 9                 # frames per glint pass
SWEEP_PASSES = 2
SWEEP_STEP = 0.040              # seconds per glint frame
SWEEP_HALF_WIDTH = 0.16         # glint half-width, in logo widths
SWEEP_ROW_OFFSET = 0.05         # diagonal tilt, so it reads as travelling light

# ---- thinking indicator -----------------------------------------------
# The splash gradient, defined once so everything that wants to look like it
# belongs can point at the same numbers: the spinner, and the input box rule.
GRADIENT_TOP = (0x22, 0xD3, 0xEE)     # cyan end
GRADIENT_BOTTOM = (0xA8, 0x55, 0xF7)  # violet end
ACCENT = GRADIENT_TOP
# ASCII frames on purpose: braille spinners are prettier but U+28xx is absent
# from a GBK codepage, and startup output already goes out of its way to survive
# a console without chcp 65001.
SPINNER_FRAMES = ("|", "/", "-", "\\")
SPINNER_INTERVAL = 0.1
SPINNER_LABEL = "思考中"

# ---- input box --------------------------------------------------------
# Two horizontal rules sandwiching the input line: no side walls, so the
# prompt reads as a line of conversation rather than a table cell.
BOX_H = "─"
BOX_H_FALLBACK = "-"
BOX_MARKER = "❯"           # as used by Claude Code, but U+276F is not in GBK
BOX_MARKER_FALLBACK = ">"
# A rule that exactly fills the last column makes some terminals emit a blank
# line of their own, so the box stops short of the edge.
BOX_MARGIN = 2
BOX_MIN_WIDTH = 40
BOX_MAX_WIDTH = 100
# Dim one-line hint under the live frame. It follows what is being typed:
# shortcuts when the line is empty, matching commands once it starts with '/',
# and the chosen command's own description once it is complete. Ordered so the
# most useful entries survive the width cut when only '/' has been typed.
BOX_HINT_COLOR = "#8a8f98"
BOX_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "全部命令"),
    ("/model", "查看/切换模型"),
    ("/theme", "改配色"),
    ("/name", "改显示名"),
    ("/status", "当前状态"),
    ("/clear", "清空上下文"),
    ("/compact", "压缩上下文"),
    ("/resume", "载入历史会话"),
    ("/forget", "删除历史会话"),
    ("/export", "导出 Markdown"),
    ("/retry", "重发上一条"),
    ("/usage", "token 用量"),
    ("/cost", "token 用量"),
    ("/doctor", "自检"),
    ("/config", "配置路径"),
    ("/skills", "已装技能"),
    ("/tools", "文件工具开关"),
    ("/dir", "工作目录"),
    ("/undo", "撤回上次修改"),
    ("/audit", "本次改动"),
    ("/persona", "角色定位"),
    ("/instructions", "附加要求"),
    ("/default-model", "设默认模型"),
    ("/key", "录入 Key"),
    ("/keys", "查看 Key"),
    ("/exit", "退出"),
    ("/quit", "退出"),
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _summarize_args(call: ToolCall) -> str:
    """One short line naming what the call is about to touch."""
    args = call.arguments or {}
    for key in ("path", "pattern"):
        if key in args:
            return str(args[key])
    return " ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _lerp_rgb(top: tuple[int, int, int], bottom: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * t) for a, b in zip(top, bottom))  # type: ignore[return-value]


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def _fg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _plain_row(line: str, base: tuple[int, int, int]) -> str:
    return f"{_fg(base)}{line}\x1b[0m"


def _sweep_row(
    line: str,
    base: tuple[int, int, int],
    phase: float,
    width: int,
    highlight: tuple[int, int, int] | None = None,
) -> str:
    """One logo row with a bright glint centred at `phase` (0..1 across the logo).

    Distance is measured around the loop rather than along the line, so the
    glint reappears on the left the instant it leaves on the right -- which is
    what makes it read as light running around the infinity instead of a
    highlight being dragged across it. `highlight` is derived from the active
    theme so a red theme does not get an icy-blue glint.
    """
    if highlight is None:
        highlight = HIGHLIGHT
    chunks: list[str] = []
    for column, char in enumerate(line):
        if char == " ":
            chunks.append(char)
            continue
        distance = abs(column / width - phase)
        distance = min(distance, 1.0 - distance)
        glow = 1.0 - distance / SWEEP_HALF_WIDTH
        colour = _blend(base, highlight, glow) if glow > 0 else base
        chunks.append(f"{_fg(colour)}{char}")
    chunks.append("\x1b[0m")
    return "".join(chunks)


def _gradient_rule(
    width: int,
    fill: str,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> str:
    """A horizontal rule whose colour runs from `start` to `end`, left to right.

    The splash gradient laid on its side, so the rules around the input are
    visibly the same object as the logo above it.
    """
    width = max(1, width)
    span = max(1, width - 1)
    parts = []
    for index in range(width):
        parts.append(f"{_fg(_blend(start, end, index / span))}{fill}")
    parts.append("\x1b[0m")
    return "".join(parts)


def _char_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def _visible_width(text: str) -> int:
    """Columns `text` occupies once escape codes are discounted.

    East-asian wide characters count as two. That is what keeps the input rules
    and the typed line the same length when the user types Chinese -- which,
    for this program, is most of the time.
    """
    return sum(_char_width(char) for char in _ANSI.sub("", text))


def _fit_hint(segments: list[tuple[str, str]], columns: int) -> list[tuple[str, str]]:
    """Cut hint fragments at the column budget, never wrapping.

    Measuring counts wide CJK glyphs, so a Chinese description cannot push the
    row past the terminal and wrap onto the conversation below.
    """
    fitted: list[tuple[str, str]] = []
    used = 0
    budget = max(0, columns)
    for style, text in segments:
        taken: list[str] = []
        for char in text:
            width = _char_width(char)
            if used + width > budget:
                if taken:
                    fitted.append((style, "".join(taken)))
                return fitted
            taken.append(char)
            used += width
        if taken:
            fitted.append((style, "".join(taken)))
    return fitted


def _command_fragments(
    matches: list[tuple[str, str]], accent: str, columns: int
) -> list[tuple[str, str]]:
    """Command suggestions as whole entries.

    A half-drawn command name ('/sta') is worse than no suggestion, so an entry
    that does not fit is dropped rather than cut. A terminal too narrow for even
    the first one still gets its name cut, since showing something beats a
    blank row.
    """
    dim = BOX_HINT_COLOR
    segments: list[tuple[str, str]] = [(dim, "  ")]
    used = 2
    for index, (name, description) in enumerate(matches):
        wanted = _visible_width(name) + 1 + _visible_width(description)
        separator = "   " if index else ""
        if used + len(separator) + wanted > columns:
            break
        if separator:
            segments.append((dim, separator))
            used += len(separator)
        segments.append((accent, name))
        segments.append((dim, f" {description}"))
        used += wanted
    if len(segments) == 1:
        return _fit_hint(
            [(dim, "  "), (accent, matches[0][0]), (dim, f" {matches[0][1]}")],
            columns,
        )
    return segments


def _hint_fragments(
    line: str, columns: int, accent: str = ""
) -> list[tuple[str, str]]:
    """The live hint under the input frame, fitted to the terminal width.

    `line` is the text before the cursor, so the hint updates with every
    keystroke: an empty box advertises the escape hatches, '/mo' narrows the
    command list to what it could still become, and a complete command explains
    itself. The indent keeps it under the typed text rather than the marker.
    One row only: the frame's height must not change as the user types, or the
    layout would jump on every keystroke.
    """
    dim = BOX_HINT_COLOR
    indent = (dim, "  ")
    columns = max(0, columns)
    line = line.split("\n", 1)[0]
    if not line:
        return _fit_hint(
            [indent, (accent, "/"), (dim, " 命令   Enter 发送   Ctrl+C 退出")],
            columns,
        )
    if not line.startswith("/"):
        return _fit_hint([indent, (dim, "Enter 发送   Ctrl+C 退出")], columns)

    word = line.split(" ", 1)[0]
    for name, description in BOX_COMMANDS:
        if name == word:
            return _fit_hint(
                [indent, (accent, name), (dim, f"  {description}   Enter 执行")],
                columns,
            )

    matches = [
        (name, description)
        for name, description in BOX_COMMANDS
        if name.startswith(word)
    ]
    if not matches:
        return _fit_hint([indent, (dim, "无匹配命令   /help 查看全部")], columns)
    return _command_fragments(matches, accent, columns)


def _take_columns(text: str, columns: int) -> str:
    """The longest prefix of `text` that fits in `columns` display columns."""
    taken: list[str] = []
    used = 0
    for char in text:
        width = _char_width(char)
        if used + width > columns:
            break
        taken.append(char)
        used += width
    return "".join(taken)


def _take_last_columns(text: str, columns: int) -> str:
    taken: list[str] = []
    used = 0
    for char in reversed(text):
        width = _char_width(char)
        if used + width > columns:
            break
        taken.append(char)
        used += width
    return "".join(reversed(taken))


def _terminal_columns(output: Any) -> int:
    """Current console width, read fresh on every layout pass.

    A value captured when the prompt opens goes stale as soon as the window is
    resized: shrink it and the rules are generated wider than the terminal, so
    they wrap and the input line no longer sits between them. prompt_toolkit
    repaints on resize (SIGWINCH, or polling on Windows), so re-reading here is
    enough to keep the rules correct.

    Output.get_size() reports the visible *window* on Windows.
    shutil.get_terminal_size() reports the screen buffer, which can be wider
    than the window, so it is only the fallback for outputs that cannot answer.
    """
    try:
        columns = int(output.get_size().columns)
        if columns > 0:
            return columns
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        return max(1, shutil.get_terminal_size().columns)
    except (OSError, ValueError):
        return 80


def _rows_below_cursor(output: Any) -> int | None:
    """Rows from the physical cursor to the bottom of the window, inclusive.

    Used to pin the input frame to the bottom when the window grows taller.
    Returns None where the output cannot answer (non-Windows terminals).
    """
    try:
        return int(output.get_rows_below_cursor_position())
    except (AttributeError, NotImplementedError, OSError, TypeError):
        return None


def _docking_extra(
    rows_below: int | None, frame_height: int, caret_row: int
) -> int:
    """Rows to step down so the frame's last row is the window's last row.

    `rows_below` counts from the caret to the bottom, inclusive. A docked caret
    sits `frame_height - 1 - caret_row` rows above the bottom: the caret's own
    row plus the rows after it (bottom rule and hint, and any input lines
    below the caret). `caret_row` is 0-based, so pasted multi-line input docks
    correctly too.
    """
    if rows_below is None:
        return 0
    target = frame_height - 1 - caret_row
    return max(0, rows_below - target)


def _bottom_plan(
    caret_row: int,
    window_top: int,
    window_bottom: int,
    buffer_height: int,
    target: int,
) -> tuple[str, int]:
    """How to put the frame's last row on the window's last row.

    `caret_row` is the caret's row in console-buffer coordinates; `target` is
    the number of rows from the caret to the frame's last row, inclusive.
    Returns one of:

    - ``("none", 0)`` -- already there, or nothing sensible can be done;
    - ``("move", top)`` -- slide the window to buffer row `top`, revealing
      more conversation above (a terminal with scrollback);
    - ``("shift", delta)`` -- copy the visible block `delta` rows down inside
      the buffer. Used when the frame is nearer the buffer's top than one
      window height, where no window position can show it at the bottom;
    - ``("repaint", 0)`` -- the rows under the caret no longer exist; a shrink
      discarded them, so the frame has to be drawn again.

    The distinction is what makes the whole visible block -- transcript and
    input frame -- move together instead of leaving the frame stranded below
    a gap.
    """
    height = window_bottom - window_top + 1
    bottom = caret_row + target - 1
    if height <= 0 or bottom < 0:
        return ("none", 0)
    if bottom > buffer_height - 1:
        return ("repaint", 0)
    if bottom == window_bottom:
        return ("none", 0)
    desired_top = bottom - height + 1
    if desired_top >= 0:
        return ("move", desired_top)
    return ("shift", window_bottom - bottom)


def _row_content_width(row: Any) -> int:
    """Columns of real content in one rendered screen row.

    prompt_toolkit pads every screen row with blanks to the full width, but
    those blanks are never written to the terminal. Only the non-space cells
    matter when working out how the terminal will rewrap the row.
    """
    return max(
        (column + char.width for column, char in row.items() if char.char != " "),
        default=0,
    )


def _rewrapped_rows_above_cursor(
    rows: list[Any], cursor_x: int, cursor_y: int, columns: int
) -> int:
    """Physical rows above the cursor after the terminal rewraps a frame.

    A shrinking window makes Windows Terminal reflow content that no longer
    fits: a long gradient rule becomes two or three rows. prompt_toolkit still
    believes the cursor is one row below the top rule, so it erases too little
    and the surplus rule rows stay behind as ghost lines. This returns the
    number of rows to move up instead: the rewrapped height of every row above
    the cursor, plus the wrapped segments of the cursor's own row that sit
    above the cursor.

    `rows` are the previous frame's screen rows and `columns` the console's new
    width. Blank padding is ignored by `_row_content_width`.
    """
    columns = max(1, columns)
    above = cursor_x // columns
    for row in rows[:cursor_y]:
        above += max(1, -(-_row_content_width(row) // columns))
    return above


def _terminal_rewraps() -> bool:
    """Whether this terminal reflows existing lines when the window narrows.

    Windows Terminal does. The legacy console host clips instead, so applying
    the rewrap arithmetic there would erase real conversation above the prompt.
    """
    return os.name == "nt" and bool(os.environ.get("WT_SESSION"))


def _reflow_columns(output: Any, size: Any) -> int:
    """The console's true column count for terminal reflow.

    Win32Output reports the window as `Right - Left`, i.e. one less than the
    real number of columns (it keeps the last column free to avoid wrapping).
    Reflow happens at the real width, and being one column short can predict an
    extra wrapped row at the boundary.
    """
    columns = int(getattr(size, "columns", 80) or 80)
    if hasattr(output, "get_win32_screen_buffer_info"):
        columns += 1
    return max(1, columns)


# ---- measuring the frame after a rewrap ------------------------------
# When the window narrows, the terminal reflows what is already on screen:
# a long rule becomes two or three physical rows. prompt_toolkit only tracks
# its logical three-row frame, so its relative "move up one row" lands in the
# middle of the wrapped rule and the rest is left behind as a ghost line.
#
# Rather than predicting the rewrap arithmetic, measure the reflowed console
# buffer (the same source conhost's own line editor uses after a resize):
# find the input row by its marker, then count the rule rows stacked above it.
# The buffer handle is opened once and reused.

_console_state: dict[str, Any] = {}


def _console_probe() -> dict[str, Any] | None:
    """Win32 console handle plus the buffer APIs, or None outside a console.

    One handle is opened once and reused. Everything here is best-effort: a
    host without these APIs (or a redirected stream) leaves the callers to
    fall back to prompt_toolkit's own cursor movement.
    """
    if os.name != "nt":
        return None
    if "probe" in _console_state:
        return _console_state["probe"]

    probe: dict[str, Any] | None = None
    try:
        import ctypes
        from ctypes import wintypes

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short),
                ("Top", ctypes.c_short),
                ("Right", ctypes.c_short),
                ("Bottom", ctypes.c_short),
            ]

        class CHAR_INFO(ctypes.Structure):
            # CHAR_INFO is a union; UnicodeChar occupies the first two bytes
            # and Attributes the next two, so this layout matches the C struct.
            _fields_ = [("Char", ctypes.c_wchar), ("Attributes", wintypes.WORD)]

        class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
            _fields_ = [
                ("dwSize", COORD),
                ("dwCursorPosition", COORD),
                ("wAttributes", wintypes.WORD),
                ("srWindow", SMALL_RECT),
                ("dwMaximumWindowSize", COORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        def declare(name: str, restype: Any, argtypes: list[Any]) -> None:
            fn = getattr(kernel32, name)
            fn.restype = restype
            fn.argtypes = argtypes

        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        handle = kernel32.CreateFileW(
            "CONOUT$", 0x80000000 | 0x40000000, 0x1 | 0x2, None, 3, 0, None
        )
        invalid = ctypes.c_void_p(-1).value
        if handle and handle != invalid and handle != -1:
            declare(
                "ReadConsoleOutputCharacterW",
                wintypes.BOOL,
                [
                    wintypes.HANDLE,
                    wintypes.LPWSTR,
                    wintypes.DWORD,
                    COORD,
                    ctypes.POINTER(wintypes.DWORD),
                ],
            )
            declare(
                "GetConsoleScreenBufferInfo",
                wintypes.BOOL,
                [wintypes.HANDLE, ctypes.POINTER(CONSOLE_SCREEN_BUFFER_INFO)],
            )
            declare(
                "SetConsoleWindowInfo",
                wintypes.BOOL,
                [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(SMALL_RECT)],
            )
            declare(
                "SetConsoleCursorPosition",
                wintypes.BOOL,
                [wintypes.HANDLE, COORD],
            )
            declare(
                "ReadConsoleOutputW",
                wintypes.BOOL,
                [
                    wintypes.HANDLE,
                    ctypes.POINTER(CHAR_INFO),
                    COORD,
                    COORD,
                    ctypes.POINTER(SMALL_RECT),
                ],
            )
            declare(
                "WriteConsoleOutputW",
                wintypes.BOOL,
                [
                    wintypes.HANDLE,
                    ctypes.POINTER(CHAR_INFO),
                    COORD,
                    COORD,
                    ctypes.POINTER(SMALL_RECT),
                ],
            )
            declare(
                "FillConsoleOutputCharacterW",
                wintypes.BOOL,
                [
                    wintypes.HANDLE,
                    ctypes.c_wchar,
                    wintypes.DWORD,
                    COORD,
                    ctypes.POINTER(wintypes.DWORD),
                ],
            )
            declare(
                "FillConsoleOutputAttribute",
                wintypes.BOOL,
                [
                    wintypes.HANDLE,
                    wintypes.WORD,
                    wintypes.DWORD,
                    COORD,
                    ctypes.POINTER(wintypes.DWORD),
                ],
            )
            probe = {
                "ctypes": ctypes,
                "kernel32": kernel32,
                "handle": handle,
                "COORD": COORD,
                "SMALL_RECT": SMALL_RECT,
                "CHAR_INFO": CHAR_INFO,
                "CSBI": CONSOLE_SCREEN_BUFFER_INFO,
                "wintypes": wintypes,
            }
    except (AttributeError, ImportError, OSError, ValueError):
        probe = None

    _console_state["probe"] = probe
    return probe


def _console_info() -> Any | None:
    """Current buffer size, window and cursor, or None when unreadable."""
    probe = _console_probe()
    if probe is None:
        return None
    try:
        info = probe["CSBI"]()
        if not probe["kernel32"].GetConsoleScreenBufferInfo(
            probe["handle"], probe["ctypes"].byref(info)
        ):
            return None
        return info
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _read_console_row(row: int, columns: int) -> str | None:
    """Text of one console-buffer row, or None when it cannot be read."""
    probe = _console_probe()
    if probe is None or row < 0 or columns <= 0:
        return None
    ctypes = probe["ctypes"]
    buffer = ctypes.create_unicode_buffer(columns + 1)
    read = probe["wintypes"].DWORD(0)
    ok = probe["kernel32"].ReadConsoleOutputCharacterW(
        probe["handle"], buffer, columns, probe["COORD"](0, row), ctypes.byref(read)
    )
    if not ok:
        return None
    return buffer[: read.value]


def _console_move_window(top: int, height: int) -> bool:
    """Slide the visible window to buffer row `top`, if the host allows it.

    A host that has no buffer rows above the window (a fresh tab, or a console
    whose buffer is only as tall as its window) may refuse, or may answer
    success while leaving the window where it was. So the result is read back
    instead of trusted.
    """
    probe = _console_probe()
    info = _console_info()
    if probe is None or info is None or top < 0 or height <= 0:
        return False
    try:
        ctypes = probe["ctypes"]
        rect = probe["SMALL_RECT"](
            int(info.srWindow.Left), top, int(info.srWindow.Right), top + height - 1
        )
        ok = probe["kernel32"].SetConsoleWindowInfo(
            probe["handle"], True, ctypes.byref(rect)
        )
        if not ok:
            return False
        after = _console_info()
        return bool(
            after is not None
            and int(after.srWindow.Top) == top
            and int(after.srWindow.Bottom) == top + height - 1
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _console_shift_down(top: int, bottom: int, delta: int, columns: int) -> bool:
    """Copy buffer rows top..bottom down by `delta`, blanking the vacated rows.

    Colour attributes travel with the cells (CHAR_INFO), so the move does not
    flatten a gradient rule or a dimmed line. Used where the window cannot be
    repositioned: sliding the drawn block down is what keeps the transcript
    glued to the input frame when the window grows.
    """
    if delta <= 0 or columns <= 0 or bottom < top:
        return True
    probe = _console_probe()
    if probe is None:
        return False
    try:
        ctypes = probe["ctypes"]
        kernel32 = probe["kernel32"]
        handle = probe["handle"]
        rows = bottom - top + 1
        size = probe["COORD"](columns, rows)
        origin = probe["COORD"](0, 0)
        cells = (probe["CHAR_INFO"] * (columns * rows))()
        source = probe["SMALL_RECT"](0, top, columns - 1, bottom)
        if not kernel32.ReadConsoleOutputW(
            handle, cells, size, origin, ctypes.byref(source)
        ):
            return False

        written = probe["wintypes"].DWORD(0)
        count = columns * rows
        kernel32.FillConsoleOutputCharacterW(
            handle, " ", count, probe["COORD"](0, top), ctypes.byref(written)
        )
        info = _console_info()
        attribute = int(info.wAttributes) if info is not None else 0x07
        kernel32.FillConsoleOutputAttribute(
            handle, attribute, count, probe["COORD"](0, top), ctypes.byref(written)
        )

        target = probe["SMALL_RECT"](0, top + delta, columns - 1, bottom + delta)
        return bool(
            kernel32.WriteConsoleOutputW(
                handle, cells, size, origin, ctypes.byref(target)
            )
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _console_set_cursor(x: int, y: int) -> None:
    """Move the physical cursor after its content has been shifted."""
    probe = _console_probe()
    if probe is None:
        return
    try:
        probe["kernel32"].SetConsoleCursorPosition(
            probe["handle"], probe["COORD"](max(0, x), max(0, y))
        )
    except (AttributeError, OSError, TypeError, ValueError):
        pass


def _measured_erase_target(
    output: Any,
    screen: Any,
    cursor_x: int,
    cursor_y: int,
    old_columns: int,
    marker: str,
    rule: str,
    line_count: int,
) -> tuple[int, int] | None:
    """Find the frame after the terminal rewrapped it, by reading the screen.

    Returns ``(caret column, rows to move up)`` or None when the console cannot
    be read or the frame cannot be identified (multi-line continuation, a
    terminal that clips instead of reflowing, or a host whose buffer is not
    reflowed). The caller then falls back to the arithmetic model.

    The input block is located by its marker: the caret may have been pushed
    onto a later wrapped segment, so walk up from the cursor until a row opens
    with ``marker + " "``. Above that row sit the wrapped copies of the top
    rule; count the contiguous run of rows whose visible content is the rule
    glyph. Both numbers are physical and therefore independent of how the
    terminal chose to wrap.
    """
    if os.name != "nt":
        return None
    try:
        info = output.get_win32_screen_buffer_info()
    except (AttributeError, OSError, TypeError):
        return None

    window = info.srWindow
    columns = int(window.Right) - int(window.Left) + 1
    if columns <= 0:
        return None
    caret_x = int(info.dwCursorPosition.X)
    caret_row = int(info.dwCursorPosition.Y)

    # A single old line was never wider than the old window; after the rewrap
    # it can occupy this many rows. That bounds the search for the input row.
    per_line = max(1, -(-max(1, old_columns) // columns))
    limit = min(caret_row, (max(1, line_count) + 1) * per_line + 2)
    marker_row: int | None = None
    for step in range(limit + 1):
        text = _read_console_row(caret_row - step, columns)
        if text is None:
            return None
        if text.startswith(marker + " "):
            marker_row = caret_row - step
            break
    if marker_row is None:
        return None

    above = caret_row - marker_row
    row_zero = screen.data_buffer[0] if getattr(screen, "height", 0) else {}
    top_width = _row_content_width(row_zero)
    max_rule_rows = max(1, -(-top_width // columns))
    for step in range(1, max_rule_rows + 1):
        text = _read_console_row(marker_row - step, columns)
        if text is None:
            break
        body = text.rstrip()
        if not body or set(body) != {rule}:
            break
        above += 1

    if above <= 0:
        return None
    return caret_x, above


def _input_box_width(content_width: int, term_width: int) -> int:
    """Outer (rule) width of the live input box.

    Tracks the typed length plus the 2 chrome columns of the "❯ " marker,
    clamped to a minimum sliver and to the terminal width minus the margin. This
    is what makes the rules widen as characters are added and narrow as they
    are deleted.

    The minimum (8) is the chrome plus a little room for the cursor, so a short
    line still shrinks toward it instead of being pinned at a wide floor.
    """
    margin = BOX_MARGIN
    min_w = 8
    return min(term_width - margin, max(min_w, content_width + 2))


def _input_text_width(box_width: int) -> int:
    """Inner width of the typing column: the rule minus the two marker cells.

    The box is sized from the longest line plus one spare cell for the caret, so
    until the terminal clamp this is the longest line plus that cell. Making the
    column exactly as wide as the text instead leaves the caret nowhere to go --
    prompt_toolkit then either wraps the line (growing the frame behind the user)
    or scrolls horizontally, clipping the first character. Past the clamp the
    column stays at its maximum and the buffer scrolls, as documented.
    """
    return max(1, box_width - 2)


def _rule_tokens(
    width: int,
    fill: str,
    top: tuple[int, int, int] = GRADIENT_TOP,
    bottom: tuple[int, int, int] = GRADIENT_BOTTOM,
) -> list[tuple[str, str]]:
    """Gradient fragments for one horizontal rule.

    Styles are built here rather than at the call site so a stray '#' cannot
    turn into an invalid '##rrggbb' token: prompt_toolkit raises on that at the
    first repaint, which aborts the takeover and silently drops the whole box
    back to the static input() fallback.
    """
    width = max(1, width)
    span = max(1, width - 1)
    return [(_hex(_blend(top, bottom, index / span)), fill) for index in range(width)]


def _compress_for_box(text: str, available: int) -> str:
    """Shorten `text` to fit `available` columns, keeping both ends.

    Both ends because a pasted block is recognised by how it opens and how it
    closes, and the middle is the part that can be spared. The count is folded
    into the marker so the user can see that the whole thing was received even
    though only part of it is on screen.
    """
    if available <= 0:
        return ""
    if _visible_width(text) <= available:
        return text

    marker = f" …共{len(text)}字… "
    if _visible_width(marker) > available:
        marker = " … "
    room = available - _visible_width(marker)
    if room <= 0:
        return _take_columns(text, available)

    head = (room * 2) // 3
    tail = room - head
    return _take_columns(text, head) + marker + _take_last_columns(text, tail)


def _enable_vt() -> bool:
    """Turn on ANSI escape processing for a Windows console.

    Rich only *detects* VT support, it never enables it, and conhost ships with
    it off -- so the cursor codes the animation needs would be printed literally
    on a plain cmd.exe. POSIX terminals speak ANSI natively, so this is a no-op
    there.

    Returns True when escape codes are safe to emit on stdout. Doing this before
    the Console is built also nudges rich onto its ANSI path, so the logo and
    the rest of the splash stay consistent.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]

        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False  # redirected, or no console at all
        if mode.value & 0x0004:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class ThinkingIndicator:
    """A one-line "still working" marker, drawn on stderr.

    stderr because it is a side channel: when stdout is redirected into a file,
    a marker must not land in it.

    It animates from a background thread because the main thread is blocked
    inside the provider's stream waiting for the first token -- which is exactly
    the silence this exists to fill. The thread is a daemon, so a hard exit can
    never hang on it.
    """

    def __init__(
        self,
        stream: TextIO,
        colour: tuple[int, int, int] = ACCENT,
        label: str = SPINNER_LABEL,
    ) -> None:
        self._stream = stream
        self._colour = _fg(colour)
        self._label = label
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._started_at = 0.0
        self._drawn = False

    def set_colour(self, colour: tuple[int, int, int]) -> None:
        self._colour = _fg(colour)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._done.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._spin, name="selfagent-thinking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Idempotent, and safe to call when it was never started."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._done.set()
        thread.join(timeout=1.0)
        with self._lock:
            self._erase()

    # ---- the thread ---------------------------------------------------

    def _spin(self) -> None:
        index = 0
        while not self._done.is_set():
            with self._lock:
                self._draw(index)
            index += 1
            self._done.wait(SPINNER_INTERVAL)

    def _draw(self, index: int) -> None:
        frame = SPINNER_FRAMES[index % len(SPINNER_FRAMES)]
        elapsed = time.monotonic() - self._started_at
        body = f"{frame} {self._label}… {elapsed:.1f}s"
        if self._write(f"\r\x1b[K{self._colour}{body}\x1b[0m"):
            self._drawn = True

    def _erase(self) -> None:
        if not self._drawn:
            return
        self._write("\r\x1b[K")
        self._drawn = False

    def _write(self, text: str) -> bool:
        try:
            self._stream.write(text)
            self._stream.flush()
            return True
        except (OSError, ValueError, AttributeError):
            # A closed or unwritable stream must never take the conversation
            # down with it; just give up on animating.
            self._done.set()
            return False


class Renderer:
    def __init__(
        self,
        *,
        color: bool = True,
        show_thinking: bool = False,
        out: TextIO | None = None,
        err: TextIO | None = None,
        animate: bool | None = None,
        spinner: bool = True,
        prompt_box: bool = True,
    ) -> None:
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.use_color = bool(color) and _isatty(self.out)
        self.show_thinking = bool(show_thinking) and self.use_color
        self.show_spinner = bool(spinner)
        self.show_box = bool(prompt_box)

        # Cosmetic defaults; cli.py replaces them once config.json is parsed.
        self.gradient_top = GRADIENT_TOP
        self.gradient_bottom = GRADIENT_BOTTOM
        self.accent = ACCENT
        self._highlight = HIGHLIGHT
        self.agent_name = "SelfAgent"

        # Whether escape codes will actually be interpreted. Resolved before the
        # Console is built so rich sees the same answer we do.
        self._vt = _enable_vt() if self.use_color else False
        # Animation is for a human watching a real console. An injected stream
        # (a test, an embedder) gets the static frame: it has no cursor to move
        # and nobody to entertain.
        #
        # Note the absence of self._vt here: a console that refuses escape codes
        # still gets the reveal, which needs none. `_animated_logo` picks the
        # variant.
        self.animate = (
            bool(animate)
            if animate is not None
            else bool(self.use_color and self.out is sys.stdout)
        )

        self._console: Any = None
        self._err_console: Any = None
        if self.use_color:
            from rich.console import Console

            # Rich reports "windows" (16 colours) on Windows no matter what the
            # terminal actually supports, which flattens the splash gradient
            # into a coarse cyan->blue. Every Windows 10+ console does 24-bit,
            # so ask for it explicitly and leave rich's detection alone
            # everywhere else.
            color_system = "truecolor" if os.name == "nt" else None
            self._console = Console(
                file=self.out,
                highlight=False,
                # No wrapping: most of what this prints is aligned, tabular text
                # -- model inventories, session listings, doctor results, diffs.
                # Wrapping one of those destroys the columns, and the terminal
                # soft-wraps anything genuinely too wide anyway.
                soft_wrap=True,
                color_system=color_system,
            )
            # Console.print() has no file= argument -- the stream is bound when
            # the Console is built. Anything headed for stderr therefore needs
            # its own Console, not a keyword on the stdout one.
            self._err_console = Console(
                file=self.err,
                highlight=False,
                soft_wrap=True,
                color_system=color_system,
            )

        self.wrote_text = False
        self._thinking_open = False
        # Where this run's own drawing starts, in console-buffer rows, and the
        # text of that first row. Used to slide the drawn block down when a
        # window cannot be moved (see _bottom_plan). Both are best-effort:
        # None means "fall back to moving the frame alone".
        self._content_top: int | None = None
        self._content_anchor: str | None = None
        # Named `_indicator` rather than anything with "thinking" in it, because
        # show_thinking already means "print the model's reasoning" -- a
        # different thing that happens to share the word.
        self._indicator = ThinkingIndicator(self.err)

    # ---- theme and identity -------------------------------------------

    def set_theme(
        self,
        top: tuple[int, int, int],
        bottom: tuple[int, int, int],
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        """Apply the decorative colours from config.json.

        Called after the config is loaded because the Renderer has to exist
        first -- it is what reports a broken config. Everything coloured here
        is cosmetic; semantic colours (warnings, errors, diffs) stay as they
        are.
        """
        self.gradient_top = top
        self.gradient_bottom = bottom
        self.accent = accent or top
        # A fixed icy glint reads as a foreign colour on a warm theme, so the
        # highlight is derived from the same gradient.
        self._highlight = _blend(_blend(top, bottom, 0.5), (255, 255, 255), 0.85)
        self._indicator.set_colour(self.accent)

    def set_agent_name(self, name: str) -> None:
        self.agent_name = (name or "").strip() or "SelfAgent"

    # ---- thinking indicator -------------------------------------------

    def _spinner_enabled(self) -> bool:
        """The marker needs a terminal to animate on and escape codes to erase
        itself with. Both are absent when output is piped -- which is also
        exactly when a stray marker would corrupt the redirected text.

        `err is sys.stderr` keeps injected streams (tests, embedders) out of it:
        they have no cursor, and spawning a thread to draw on them would only
        make tests flaky.
        """
        return bool(
            self.show_spinner
            and self.use_color
            and self._vt
            and self.err is sys.stderr
        )

    def thinking_start(self) -> None:
        if self._spinner_enabled():
            self._indicator.start()

    def thinking_stop(self) -> None:
        self._indicator.stop()

    # ---- stream events ------------------------------------------------

    def handle(self, event: StreamEvent) -> None:
        if isinstance(event, ThinkingDelta):
            self._write_thinking(event.text)
        elif isinstance(event, TextDelta):
            self._write_text(event.text)
        # Usage and Done are bookkeeping; the caller consumes them directly.

    def _write_text(self, text: str) -> None:
        if not text:
            return
        self.thinking_stop()
        self._close_thinking()
        self.wrote_text = True
        if self._console is not None:
            self._console.out(text, end="")
        else:
            self.out.write(text)
            self.out.flush()

    def _write_thinking(self, text: str) -> None:
        if not self.show_thinking or not text:
            return
        self.thinking_stop()
        if not self._thinking_open:
            self._thinking_open = True
            self._console.print("[dim]── 思考 ──[/dim]")
        self._console.out(text, style="dim", end="")

    def _close_thinking(self) -> None:
        if self._thinking_open:
            self._thinking_open = False
            self._console.print()

    # ---- lifecycle ----------------------------------------------------

    def finish(self) -> None:
        """Close any open block and end the line. Safe to call repeatedly."""
        self.thinking_stop()
        self._close_thinking()
        if self.wrote_text and self.use_color:
            self._console.print()
        elif self.wrote_text:
            self.out.write("\n")
        self.out.flush()
        self.wrote_text = False

    # ---- splash -------------------------------------------------------

    def _animating(self) -> bool:
        """Animation needs a console to draw on; whether it can use cursor codes
        is a separate question, answered inside `_animated_logo`."""
        return bool(self.animate and self.use_color and self._console is not None)

    def _emit_logo_line(self, line: str, colour: str) -> None:
        if self._console is not None:
            self._console.print(
                line, style=colour, markup=False, highlight=False, soft_wrap=True
            )
        else:
            self.out.write(line + "\n")

    def _static_logo(self, lines: list[str], top, bottom) -> None:
        span = max(1, len(lines) - 1)
        for index, line in enumerate(lines):
            self._emit_logo_line(line, _hex(_lerp_rgb(top, bottom, index / span)))

    def _reveal_logo(self, lines: list[str], top, bottom) -> None:
        """Rows appear one at a time.

        Needs no cursor control, so this is what runs on a console whose VT
        processing could not be switched on: less flashy, still alive.
        """
        span = max(1, len(lines) - 1)
        for index, line in enumerate(lines):
            self._emit_logo_line(line, _hex(_lerp_rgb(top, bottom, index / span)))
            self.out.flush()
            time.sleep(REVEAL_STEP)

    def _animated_logo(self, lines: list[str], top, bottom, highlight) -> None:
        """Reveal the logo, run a glint along it, then settle.

        The final frame is the plain gradient, so the screen ends up exactly
        where the static splash would have left it -- the animation decorates
        the logo, it does not replace it.
        """
        if not lines:
            return
        if not self._vt:
            self._reveal_logo(lines, top, bottom)
            return

        width = max(1, max(len(line) for line in lines))
        span = max(1, len(lines) - 1)
        bases = [_lerp_rgb(top, bottom, index / span) for index in range(len(lines))]

        frames: list[tuple[list[str], float]] = []
        for shown in range(1, len(lines) + 1):
            frames.append(
                ([_plain_row(lines[r], bases[r]) for r in range(shown)], REVEAL_STEP)
            )
        for _pass in range(SWEEP_PASSES):
            for step in range(SWEEP_STEPS):
                phase = step / SWEEP_STEPS
                frames.append(
                    (
                        [
                            _sweep_row(
                                lines[r],
                                bases[r],
                                (phase + r * SWEEP_ROW_OFFSET) % 1.0,
                                width,
                                highlight,
                            )
                            for r in range(len(lines))
                        ],
                        SWEEP_STEP,
                    )
                )
        frames.append(([_plain_row(lines[r], bases[r]) for r in range(len(lines))], 0.0))

        self.out.flush()
        self.out.write("\x1b[?25l")  # hide the cursor, or it flickers over the art
        drawn = 0
        try:
            for rows, delay in frames:
                # CPL moves to column 1 of the line `drawn` rows up, so every
                # redraw starts from the same corner and the logo stays put.
                buffer = [f"\x1b[{drawn}F"] if drawn else []
                for row in rows:
                    buffer.append("\x1b[2K")  # erase first: rows can shorten
                    buffer.append(row)
                    buffer.append("\n")
                self.out.write("".join(buffer))
                self.out.flush()
                drawn = len(rows)
                if delay:
                    time.sleep(delay)
        finally:
            self.out.write("\x1b[?25h")
            self.out.flush()

    def _encodable(self, text: str) -> bool:
        encoding = getattr(self.out, "encoding", None) or "utf-8"
        try:
            text.encode(encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    def splash(
        self,
        logo: tuple[str, ...],
        info: list[tuple[str, str]],
        *,
        top: tuple[int, int, int] | None = None,
        bottom: tuple[int, int, int] | None = None,
    ) -> None:
        """Startup screen: gradient logo, then styled info lines.

        `info` is (text, rich style) pairs so the caller can mark up which
        providers are usable without this module knowing what a provider is.
        Colours default to the theme applied by set_theme().

        Always drawn, unlike status(): the REPL only runs interactively, so this
        is never the machine-read output that status() protects.

        The logo is skipped for a console whose codepage cannot carry the block
        glyphs. Crashing at startup on an unfamiliar machine is the one failure
        this tool exists to avoid, and a degraded logo costs nothing.
        """
        top = self.gradient_top if top is None else top
        bottom = self.gradient_bottom if bottom is None else bottom

        # Remember where this run starts drawing. If a window later grows and
        # the host cannot move its viewport, everything from this row down is
        # what slides to stay glued to the input frame.
        console = _console_info()
        self._content_top = (
            int(console.dwCursorPosition.Y) if console is not None else None
        )
        self._content_anchor = None

        lines = [line.rstrip() for line in logo]
        if not self._encodable("".join(lines)):
            fallback = f"∞ {self.agent_name}" if self._encodable("∞") else self.agent_name
            lines = [fallback]

        if self._animating():
            highlight = _blend(_blend(top, bottom, 0.5), (255, 255, 255), 0.85)
            self._animated_logo(lines, top, bottom, highlight)
        else:
            self._static_logo(lines, top, bottom)

        for text, style in info:
            if self._console is not None:
                self._console.print(
                    text, style=style or None, markup=False, highlight=False
                )
            else:
                self.out.write(text + "\n")

        self.out.flush()
        # The first drawn row identifies where this block begins. If the buffer
        # later scrolls, that row holds something else and the anchor lets the
        # shift fall back to the top of the visible window instead of dropping
        # a blank band into the middle of the conversation.
        if self._content_top is not None:
            console = _console_info()
            if console is not None:
                self._content_anchor = _read_console_row(
                    self._content_top, max(1, int(console.dwSize.X))
                )

    # ---- input box ----------------------------------------------------

    def _box_enabled(self) -> bool:
        """The box is drawn with escape codes, so it needs a console that
        interprets them. Without one the caller gets the plain prompt instead --
        a box made of literal `\\x1b[38;2;...` text is worse than no box.
        """
        return bool(self.show_box and self.use_color and self._vt and self.out is sys.stdout)

    def _box_chars(self) -> dict[str, str]:
        """Horizontal rule and marker glyphs, degrading to ASCII."""
        if self._encodable(BOX_H + BOX_MARKER):
            return {"h": BOX_H, "marker": BOX_MARKER}
        marker = BOX_MARKER if self._encodable(BOX_MARKER) else BOX_MARKER_FALLBACK
        if self._encodable(BOX_H):
            return {"h": BOX_H, "marker": marker}
        return {"h": BOX_H_FALLBACK, "marker": BOX_MARKER_FALLBACK}

    def _box_width(self) -> int:
        detected = getattr(self._console, "width", 80) or 80
        return max(BOX_MIN_WIDTH, min(BOX_MAX_WIDTH, detected - BOX_MARGIN))

    def _prompt_toolkit(self, label: str) -> str | None:
        """Take over the terminal for a growing, IME-safe input box.

        Returns the entered text, or None when prompt_toolkit cannot be used on
        this console (not installed, not a real TTY, or it fails to start) so the
        caller can fall back to the static input() box. The static box stays the
        guaranteed-available path; this is the upgrade.

        prompt_toolkit is imported lazily on purpose: it must never be a hard
        import, or a machine without it (or one where the frozen bundle omitted
        it) would fail to even load this module. A hand-rolled raw reader was
        rejected because it breaks Windows IME composition -- the input method
        this program exists for.
        """
        if self.out is not sys.stdout:
            return None
        try:
            import prompt_toolkit  # noqa: F401  -- proves it is importable
        except Exception:
            return None
        if not (getattr(sys.stdin, "isatty", lambda: False)() and sys.stdout.isatty()):
            return None
        try:
            return self._run_pt_session()
        except (EOFError, KeyboardInterrupt):
            # Quit signals must reach the REPL, not be swallowed into a fallback.
            raise
        except Exception:
            # The frame may already be half-drawn; step off it, then degrade.
            try:
                self.out.write("\n")
            except Exception:
                pass
            return None

    def _run_pt_session(self) -> str:
        """Draw an input line between two gradient rules that follow the text.

        The terminal is taken over by prompt_toolkit so the rules are repainted
        on every keystroke: they widen as characters are added and narrow as
        they are deleted, down to a minimum and up to the terminal width (beyond
        that the text scrolls). A pasted block adds one row per line, each
        aligned under the "❯ " marker, and the rows are given back when the
        lines are deleted. This is what the static input() box could never do --
        input() hands the screen to the OS and cannot repaint mid-input. No side
        walls: the two rules plus one dim command hint below them.
        """
        from prompt_toolkit import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.data_structures import Point
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout.containers import (
            ConditionalContainer,
            HSplit,
            VSplit,
            Window,
        )
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.layout import Layout
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.renderer import Renderer as _PTRenderer

        chars = self._box_chars()
        buffer = Buffer(multiline=False)

        def lines() -> list[str]:
            return buffer.text.split("\n")

        def content_width() -> int:
            # East-asian wide characters count as two, just like the rest of the
            # box maths, so a Chinese line and a Latin line of equal "feel" widen
            # the box by the same real amount. The LONGEST line decides the
            # width: a pasted block's newlines are not columns.
            return max((_visible_width(line) for line in lines()), default=0)

        def box_width() -> int:
            # +1 is the cell the caret occupies after the last character. Without
            # it the text column is exactly as wide as the text and the caret has
            # nowhere to go. The terminal width is read live: the window can be
            # resized while the prompt is open.
            return _input_box_width(content_width() + 1, _terminal_columns(app.output))

        def text_area_width() -> int:
            return _input_text_width(box_width())

        def top_text() -> list[tuple[str, str]]:
            return _rule_tokens(
                box_width(), chars["h"], self.gradient_top, self.gradient_bottom
            )

        def bottom_text() -> list[tuple[str, str]]:
            return _rule_tokens(
                box_width(), chars["h"], self.gradient_top, self.gradient_bottom
            )

        def marker_fragments() -> list[tuple[str, str]]:
            # One marker per document line. A pasted block makes the row several
            # lines tall; continuation rows get the same two-column indent so
            # they line up under the typed text.
            first = chars["marker"] + " "
            rest = ["  "] * (len(lines()) - 1)
            return [(_hex(self.accent), "\n".join([first] + rest))]

        top_rule = Window(FormattedTextControl(top_text), height=1)
        marker = Window(FormattedTextControl(marker_fragments), width=2)
        text_area = Window(
            BufferControl(buffer),
            # A callable width is re-evaluated on every layout pass, so the text
            # column shrinks and grows with the input. prompt_toolkit resolves
            # the callable itself (Dimension() would not accept one).
            width=text_area_width,
            # No soft wrapping: the rules grow to the terminal and then the text
            # scrolls. Wrapping would add visual rows below the marker line,
            # which the rules control knows nothing about.
            wrap_lines=False,
            # The horizontal scroll offset otherwise survives across lines: after
            # a long line, moving the caret to a short one makes prompt_toolkit
            # scroll the window to the caret column and hide that line's own
            # text. Recomputing from zero every frame still follows the caret
            # right when the line is longer than the window.
            get_horizontal_scroll=lambda _window: 0,
        )
        bottom_rule = Window(FormattedTextControl(bottom_text), height=1)

        def hint_fragments() -> list[tuple[str, str]]:
            # Reads the buffer on every render, and every keystroke invalidates
            # the app, so this is re-evaluated as the user types. The width is
            # re-read live for the same reason the rules are.
            return _hint_fragments(
                buffer.document.current_line_before_cursor,
                _terminal_columns(app.output),
                _hex(self.accent),
            )

        # Hidden just before the prompt closes. prompt_toolkit's final render
        # then has one row fewer, and its erase-down clears the hint so the
        # conversation below starts clean.
        hint_on = {"show": True}
        hint = ConditionalContainer(
            content=Window(FormattedTextControl(hint_fragments), height=1),
            filter=Condition(lambda: hint_on["show"]),
        )

        root = HSplit([
            top_rule,
            VSplit([marker, text_area]),
            bottom_rule,
            hint,
        ])

        kb = KeyBindings()
        @kb.add("enter")
        def _accept(event) -> None:
            hint_on["show"] = False
            event.app.exit(result=buffer.text)
        @kb.add("c-c")
        def _ctrl_c(event) -> None:
            hint_on["show"] = False
            raise KeyboardInterrupt
        @kb.add("c-d")
        def _ctrl_d(event) -> None:
            hint_on["show"] = False
            raise EOFError

        # Construct first: without a real console this raises, and the caller's
        # try/except falls back to the input() box. Nothing is drawn yet.
        renderer = self

        class _App(Application):
            """Repaint on width changes and keep the frame at the window bottom.

            The frame is inline, but a taller window adds blank rows below the
            cursor. `_sync_to_bottom` keeps the frame's last row on the window's
            last row -- and, more importantly, keeps the drawn block above it
            attached: where the host has scrollback the window slides so more
            conversation shows above; where it does not, the block itself is
            copied down and the vacated rows are blanked. Either way the
            transcript and the input move together, the way Claude's terminal
            UI does. Column changes are separate: the terminal rewraps the old
            rules, so the frame is erased and redrawn at the new width.
            """

            def _on_resize(self) -> None:
                size = self.output.get_size()
                last = self.renderer._last_size
                if last is None or size.columns == last.columns:
                    return
                self._refit()

            def _sync_to_bottom(self, preferred: int, caret_row: int) -> None:
                """Anchor the frame's last row to the window's last row."""
                target = preferred - 1 - caret_row
                info = _console_info()
                if info is None:
                    # No Win32 console access: the old frame-only move is the
                    # best available approximation.
                    self._move_to_bottom(preferred, caret_row)
                    return

                caret_y = int(info.dwCursorPosition.Y)
                window = info.srWindow
                window_top = int(window.Top)
                window_bottom = int(window.Bottom)
                height = window_bottom - window_top + 1
                buffer_height = int(info.dwSize.Y)
                columns = max(1, int(info.dwSize.X))
                action, value = _bottom_plan(
                    caret_y, window_top, window_bottom, buffer_height, target
                )
                if action == "none":
                    return
                if action == "move":
                    if _console_move_window(value, height):
                        return
                    # The host refused to move its viewport; fall through and
                    # slide the block instead (a fresh tab is the common case).
                if action in ("move", "shift"):
                    bottom = caret_y + target - 1
                    delta = window_bottom - bottom
                    source = self._shift_source(window_top, bottom, columns)
                    if delta > 0 and _console_shift_down(
                        source, bottom, delta, columns
                    ):
                        _console_set_cursor(
                            int(info.dwCursorPosition.X), caret_y + delta
                        )
                        if (
                            renderer._content_top is not None
                            and source == renderer._content_top
                        ):
                            renderer._content_top += delta
                        return
                # The rows under the caret were destroyed by a shrink, or the
                # block could not be moved: repaint and let the renderer scroll
                # the frame back into view.
                self._refit()

            def _shift_source(self, window_top: int, bottom: int, columns: int) -> int:
                """First row of the block to slide down.

                Normally the start of this run's own drawing, so the shell's
                prompt above it is left alone. If that row no longer holds the
                text recorded at startup, the buffer has scrolled: the block
                above is already whatever is visible, so start at the window top
                rather than dropping a blank band into the middle of it.
                """
                top = renderer._content_top
                if top is None or top < window_top or top > bottom:
                    return window_top
                anchor = renderer._content_anchor
                if anchor:
                    if _read_console_row(top, columns) != anchor:
                        return window_top
                return top

            def _move_to_bottom(self, preferred: int, caret_row: int) -> None:
                """Move only the frame down until its last row is the bottom one.

                The fallback for outputs that cannot report console geometry.
                """
                rows_below = _rows_below_cursor(self.output)
                extra = _docking_extra(rows_below, preferred, caret_row)
                if extra <= 0:
                    return
                self.renderer.erase(leave_alternate_screen=False)
                self.output.cursor_down(extra)
                self.output.flush()
                self._request_absolute_cursor_position()
                self._redraw()

            def _refit(self) -> None:
                """Repaint in place after a shrink.

                The terminal may have clipped the bottom rule and hint, or
                scrolled the content. Erasing from the caret and redrawing lets
                the renderer's own reservation scroll the frame back into view,
                instead of leaving it half off-screen.
                """
                self.renderer.erase(leave_alternate_screen=False)
                self._request_absolute_cursor_position()
                self._redraw()

        app = _App(
            layout=Layout(root),
            key_bindings=kb,
            full_screen=False,  # draw inline, just below the cursor
            mouse_support=False,
            # The default poll is 0.5s; prompt_toolkit has no resize signal on
            # Windows, so that is how long a resized frame stays stale.
            terminal_size_polling_interval=0.15,
        )

        class _RepaintRenderer(_PTRenderer):
            """Repaint the whole frame whenever the column count changes.

            Inline prompt_toolkit normally patches the previous screen cell by
            cell, using relative cursor moves. That is cheap, but it assumes
            the terminal has not touched the rows underneath. Resizing breaks
            the assumption: a narrower window rewraps the old rules, so the
            logical cursor no longer matches the physical one.

            So on any width change the previous screen is dropped and the
            frame is redrawn from scratch -- the same whole-frame repaint a
            React-style renderer does, without giving up the line editor.
            (This also gives rows back after a pasted block shrinks: inline
            rendering otherwise keeps at least the previous height.)
            """

            def render(self, app, layout, is_done: bool = False) -> None:
                size = self.output.get_size()
                preferred = layout.container.preferred_height(
                    size.columns, size.rows
                ).preferred
                if not is_done:
                    if self._last_screen is not None:
                        # A keystroke can arrive before the resize poller
                        # fires. Repaint here, while the previous screen is
                        # still known, instead of patching a frame that the
                        # terminal has already rewrapped.
                        if (
                            self._last_size is not None
                            and size.columns != self._last_size.columns
                        ):
                            self.erase(leave_alternate_screen=False)
                        elif preferred < self._last_screen.height:
                            self._last_screen = None
                    # Reserve only the frame, not the rest of the window.
                    # prompt_toolkit otherwise treats the whole remaining
                    # height as its canvas and jumps the cursor to the bottom
                    # of it on every repaint; that jump scrolls the terminal,
                    # and after un-maximizing the frame can start one row lower,
                    # leaving the old top rule stranded above it.
                    self._min_available_height = min(
                        self._min_available_height, preferred
                    )
                super().render(app, layout, is_done)

            def erase(self, leave_alternate_screen: bool = True) -> None:
                # A resize can rewrap the old frame before we hear about it,
                # so the top rule may now sit several rows above the cursor.
                # Erasing from the cursor's remembered row would leave those
                # wrapped copies on screen as ghost rules. Measure the real
                # frame first; reset() then drops the previous screen so the
                # next render is a full repaint rather than a diff.
                if (
                    self._last_screen is not None
                    and self._last_size is not None
                    and self._cursor_pos is not None
                ):
                    size = self.output.get_size()
                    if size.columns != self._last_size.columns:
                        target = _measured_erase_target(
                            self.output,
                            self._last_screen,
                            self._cursor_pos.x,
                            self._cursor_pos.y,
                            self._last_size.columns,
                            chars["marker"],
                            chars["h"],
                            len(lines()),
                        )
                        if target is None and _terminal_rewraps():
                            # The screen cannot be measured (or a host whose
                            # buffer is not reflowed): predict from the old
                            # frame instead.
                            rows = [
                                self._last_screen.data_buffer[y]
                                for y in range(self._last_screen.height)
                            ]
                            columns = _reflow_columns(self.output, size)
                            above = _rewrapped_rows_above_cursor(
                                rows,
                                self._cursor_pos.x,
                                self._cursor_pos.y,
                                columns,
                            )
                            target = (self._cursor_pos.x % columns, above)
                        if target is not None:
                            self._cursor_pos = Point(x=target[0], y=target[1])
                super().erase(leave_alternate_screen)

        app.renderer = _RepaintRenderer(
            app.renderer.style,
            app.output,
            full_screen=False,
            mouse_support=False,
            cpr_not_supported_callback=app.cpr_not_supported_callback,
        )

        def _start_following() -> None:
            # A timer instead of prompt_toolkit's resize callback: on Windows
            # the reported size changes can be coalesced while the window edge
            # is dragged, which made the frame move only once. The timer looks
            # at the real distance to the bottom, so every blank row that
            # appears is reclaimed as soon as it is visible. Following starts
            # at the first height change (a short conversation is not pushed
            # down on startup) and then stays on for the rest of the prompt:
            # shrinking the window must bring the frame back, not switch
            # following off.
            import asyncio

            async def follow() -> None:
                previous_rows: int | None = None
                docking = False
                stalls = 0
                while not app.is_done:
                    await asyncio.sleep(0.05)
                    try:
                        size = app.output.get_size()
                        resized = (
                            previous_rows is not None
                            and size.rows != previous_rows
                        )
                        if resized:
                            docking = True
                            stalls = 0
                        previous_rows = size.rows
                        if not docking or stalls >= 4:
                            continue

                        preferred = app.layout.container.preferred_height(
                            size.columns, size.rows
                        ).preferred
                        caret_row = buffer.document.cursor_position_row
                        target = preferred - 1 - caret_row
                        rows_below = _rows_below_cursor(app.output)
                        if rows_below is None:
                            continue
                        if rows_below == target:
                            stalls = 0
                            continue
                        app._sync_to_bottom(preferred, caret_row)
                        # Verify instead of assuming: a host that silently
                        # refuses to move must not be hammered every tick.
                        rows_below = _rows_below_cursor(app.output)
                        stalls = 0 if rows_below == target else stalls + 1
                    except Exception:
                        # A repaint problem must not kill the follower; the next
                        # tick retries after the screen state has settled.
                        continue

            app.create_background_task(follow())

        return app.run(pre_run=_start_following)

    def prompt(self, label: str = "你") -> str:
        """Read one line (or a pasted block) inside a gradient-ruled box.

        When prompt_toolkit is available and the console is real, the terminal is
        taken over so the frame can grow with pasted multi-line text and Chinese
        IME composition keeps working. Otherwise the static input()-based box
        below is drawn -- the same fallback that runs on a console where the
        takeover cannot be set up. Either way a bare `你> ` never appears where a
        box can be.

        Raises EOFError / KeyboardInterrupt so the REPL knows to quit.
        """
        if not self._box_enabled():
            return input(f"\n{label}> ")

        line = self._prompt_toolkit(label)
        if line is not None:
            return line

        # --- static box (fallback when prompt_toolkit is unavailable) -----
        chars = self._box_chars()
        width = self._box_width()

        def rule() -> str:
            return _gradient_rule(
                width,
                chars["h"],
                self.gradient_top,
                self.gradient_bottom,
            )

        prompt_text = f"\x1b[1m{_fg(self.accent)}{chars['marker']}\x1b[0m "
        padding = " " * max(0, width - _visible_width(prompt_text))

        self.out.write("\n")
        self.out.write(rule() + "\n")
        self.out.write(f"{prompt_text}{padding}\n")
        self.out.write(rule() + "\n")
        # Two rows back up, then across to just past the marker. Only the cursor
        # moves; the box is already finished and stays that way.
        self.out.write(f"\x1b[2A\x1b[{_visible_width(prompt_text) + 1}G")
        self.out.flush()

        try:
            line = input("")
        except BaseException:
            # Steps past the box even on Ctrl+C or Ctrl+D, so the REPL's parting
            # newline does not land inside the frame.
            self.out.write("\n")
            self.out.flush()
            raise

        self._settle_input(chars, width, prompt_text, line, rule)
        return line

    def _input_rows(self, prompt_width: int, text_width: int) -> int:
        """How many visual rows the echoed input occupied.

        The console does the echoing, so this can only be worked out afterwards:
        a row holds the terminal's width in columns, and the first row starts
        part-way across because of the prompt.
        """
        terminal = max(1, getattr(self._console, "width", 80) or 80)
        first_row = max(1, terminal - prompt_width)
        if text_width <= first_row:
            return 1
        return 1 + -(-(text_width - first_row) // terminal)

    def _settle_input(self, chars, width, prompt_text, line, rule) -> None:
        """Tidy up after a line that was too long for the box.

        The console echoes what is typed, so an over-long paste cannot be
        prevented -- only cleaned up once it has happened. Best effort by
        design: if the row arithmetic is wrong on some terminal the result is
        untidy rather than broken, and the full text is submitted either way.
        """
        prompt_width = _visible_width(prompt_text)
        text_width = _visible_width(line)

        if prompt_width + text_width <= width - 1:
            # The bottom rule drawn before input is still intact.
            self.out.write("\n")
            self.out.flush()
            return

        rows = self._input_rows(prompt_width, text_width)
        shown = _compress_for_box(line, max(1, width - prompt_width - 1))

        self.out.write(f"\x1b[{rows}A\r")
        self.out.write(f"{prompt_text}{shown}\x1b[0K")
        for _ in range(rows - 1):
            self.out.write("\n\x1b[2K")
        self.out.write("\n")
        self.out.write(rule() + "\n")
        self.out.flush()

    # ---- tool activity ------------------------------------------------

    def tool_start(self, call: ToolCall) -> None:
        """Announce a tool call before it runs.

        Goes to stderr: this is a side-channel record of what the model is
        doing to the filesystem, and must not contaminate the answer when
        stdout is redirected into a file or a pipe.
        """
        detail = _summarize_args(call)
        self.thinking_stop()
        if self._err_console is not None:
            accent = _hex(self.accent)
            self._err_console.print(
                f"[{accent}]· {call.name}[/{accent}] [dim]{detail}[/dim]"
            )
        else:
            print(f"· {call.name} {detail}", file=self.err, flush=True)

    def tool_end(self, result: ToolResult) -> None:
        if not result.is_error:
            return
        self.thinking_stop()
        first = (result.content.strip().splitlines() or ["失败"])[0]
        if self._err_console is not None:
            self._err_console.print(f"[yellow]  {first}[/yellow]")
        else:
            print(f"  {first}", file=self.err, flush=True)

    def diff(self, action: str, path: Path, old: str, new: str) -> None:
        """Show what a write would change, before asking for approval."""
        self.thinking_stop()
        lines = list(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"{path} (原)",
                tofile=f"{path} (新)",
                lineterm="",
            )
        )

        header = f"模型请求{action}:{path}"
        if self._err_console is not None:
            self._err_console.print(f"[bold]{header}[/bold]")
        else:
            print(header, file=self.err, flush=True)

        if not lines:
            note = "  (内容无变化)"
            if self._err_console is not None:
                self._err_console.print(f"[dim]{note}[/dim]")
            else:
                print(note, file=self.err, flush=True)
            return

        shown = lines[:MAX_DIFF_LINES]
        for line in shown:
            style = ""
            if line.startswith("+++") or line.startswith("---"):
                style = "dim"
            elif line.startswith("+"):
                style = "green"
            elif line.startswith("-"):
                style = "red"
            elif line.startswith("@@"):
                style = _hex(self.accent)
            if self._err_console is not None:
                self._err_console.print(line, style=style or None, markup=False)
            else:
                print(line, file=self.err, flush=True)

        if len(lines) > MAX_DIFF_LINES:
            more = f"  … 差异较长,只显示前 {MAX_DIFF_LINES} 行(共 {len(lines)} 行)"
            if self._err_console is not None:
                self._err_console.print(f"[dim]{more}[/dim]")
            else:
                print(more, file=self.err, flush=True)

    def ask_permission(self, question: str) -> bool:
        """Blocking y/N. Only ever called on an interactive terminal."""
        try:
            answer = input(f"{question} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer.strip().lower() in ("y", "yes")

    # ---- side channels ------------------------------------------------

    def status(self, text: str) -> None:
        """Informational line. Goes to stdout only when a human is watching."""
        self.thinking_stop()
        if self._console is not None:
            self._console.print(f"[dim]{text}[/dim]")
        # Deliberately silent when piped: status chatter would corrupt the
        # machine-readable output the redirect is asking for.

    def warning(self, text: str) -> None:
        """Always visible, and on stderr so it cannot corrupt piped stdout."""
        self.thinking_stop()
        if self._err_console is not None:
            self._err_console.print(f"[yellow]! {text}[/yellow]")
        else:
            print(f"! {text}", file=self.err, flush=True)

    def error(self, text: str) -> None:
        self.thinking_stop()
        if self._err_console is not None:
            self._err_console.print(f"[red]错误:{text}[/red]")
        else:
            print(f"错误:{text}", file=self.err, flush=True)


def read_secret(label: str) -> str:
    """Read a secret from the console with echo off.

    Three things this does that a bare getpass(label) does not:

    - Writes the label through our own stderr instead of handing it to
      getpass. On Windows getpass emits it with msvcrt.putwch straight into
      the console buffer, bypassing the streams entirely, and it can be
      missed while a just-launched process is still settling -- the user then
      sees a bare cursor and cannot tell the program is waiting.
    - Drops whatever is already queued in the console input buffer. A stray
      Enter -- a double tap, or a pasted command whose trailing newline the
      shell did not consume -- arrives as an empty password, and the caller
      then reports a cancel nobody asked for. Everything queued before the
      prompt is drawn is stale by definition: the user answers after seeing it.
    - Treats Enter-on-empty and Ctrl+C as a cancel, same as before.
    """
    import getpass

    _drain_pending_keys()
    if sys.stderr is not None:
        sys.stderr.write(label)
        sys.stderr.flush()
    try:
        return getpass.getpass("")
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _drain_pending_keys() -> None:
    try:
        import msvcrt
    except ImportError:
        return
    try:
        while msvcrt.kbhit():
            msvcrt.getwch()
    except OSError:
        pass


def summarize_usage(usage: Usage) -> str | None:
    if not usage.input_tokens and not usage.output_tokens:
        return None
    return f"tokens: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}"
