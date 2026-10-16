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
import sys
import threading
import time
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
# One colour lifted from the splash gradient -- its cyan end -- so the spinner
# and the logo read as the same program rather than two unrelated things.
ACCENT = (0x22, 0xD3, 0xEE)
# ASCII frames on purpose: braille spinners are prettier but U+28xx is absent
# from a GBK codepage, and startup output already goes out of its way to survive
# a console without chcp 65001.
SPINNER_FRAMES = ("|", "/", "-", "\\")
SPINNER_INTERVAL = 0.1
SPINNER_LABEL = "思考中"


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


def _sweep_row(line: str, base: tuple[int, int, int], phase: float, width: int) -> str:
    """One logo row with a bright glint centred at `phase` (0..1 across the logo).

    Distance is measured around the loop rather than along the line, so the
    glint reappears on the left the instant it leaves on the right -- which is
    what makes it read as light running around the infinity instead of a
    highlight being dragged across it.
    """
    chunks: list[str] = []
    for column, char in enumerate(line):
        if char == " ":
            chunks.append(char)
            continue
        distance = abs(column / width - phase)
        distance = min(distance, 1.0 - distance)
        glow = 1.0 - distance / SWEEP_HALF_WIDTH
        colour = _blend(base, HIGHLIGHT, glow) if glow > 0 else base
        chunks.append(f"{_fg(colour)}{char}")
    chunks.append("\x1b[0m")
    return "".join(chunks)


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
    ) -> None:
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self.use_color = bool(color) and _isatty(self.out)
        self.show_thinking = bool(show_thinking) and self.use_color
        self.show_spinner = bool(spinner)

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
        # Named `_indicator` rather than anything with "thinking" in it, because
        # show_thinking already means "print the model's reasoning" -- a
        # different thing that happens to share the word.
        self._indicator = ThinkingIndicator(self.err)

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

    def _animated_logo(self, lines: list[str], top, bottom) -> None:
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
        top: tuple[int, int, int] = (0x22, 0xD3, 0xEE),
        bottom: tuple[int, int, int] = (0xA8, 0x55, 0xF7),
    ) -> None:
        """Startup screen: gradient logo, then styled info lines.

        `info` is (text, rich style) pairs so the caller can mark up which
        providers are usable without this module knowing what a provider is.

        Always drawn, unlike status(): the REPL only runs interactively, so this
        is never the machine-read output that status() protects.

        The logo is skipped for a console whose codepage cannot carry the block
        glyphs. Crashing at startup on an unfamiliar machine is the one failure
        this tool exists to avoid, and a degraded logo costs nothing.
        """
        lines = [line.rstrip() for line in logo]
        if not self._encodable("".join(lines)):
            fallback = "∞ SelfAgent" if self._encodable("∞") else "SelfAgent"
            lines = [fallback]

        if self._animating():
            self._animated_logo(lines, top, bottom)
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
            self._err_console.print(
                f"[cyan]· {call.name}[/cyan] [dim]{detail}[/dim]"
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
                style = "cyan"
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
