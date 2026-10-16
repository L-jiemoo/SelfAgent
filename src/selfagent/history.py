"""Reading the conversation log back in.

Session writes the log one JSON object per line, best-effort, and it is the only
thing that outlives a run. Resume is therefore a reader for it -- and it has to
cope with logs it did not write:

* A USB stick pulled mid-write leaves a half-finished final line.
* Logs from before ids were recorded carry tool calls without them, and both
  APIs require a tool call to be answered by a result with the *same* id.

The second one is the subtle part. Pairing is positional because the session
model guarantees an assistant entry carrying N calls is immediately followed by
a user entry carrying N results, so ids can be rebuilt for old logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .providers.base import Message, ToolCall, ToolResult
from .session import COMPACT_MARKER

PREVIEW_CHARS = 40
MAX_LIST = 20
LOOSE_OPENING = "(接续此前的对话)"


@dataclass
class SessionSummary:
    path: Path
    started: str
    target: str
    turns: int
    preview: str


def _records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A log cut off mid-write is expected, not exceptional. Everything
            # before the damage is still perfectly usable.
            continue
        if isinstance(record, dict):
            yield record


def _one_line(text: Any) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) > PREVIEW_CHARS:
        return collapsed[: PREVIEW_CHARS - 1] + "…"
    return collapsed


def summarise(path: Path) -> SessionSummary | None:
    """Read one log's header and shape. None when it is not one of ours."""
    started = ""
    target = ""
    turns = 0
    preview = ""

    for record in _records(path):
        kind = record.get("type")
        if kind == "session_start":
            started = str(record.get("time", ""))
            target = str(record.get("target", ""))
        elif kind == "user" and not preview:
            preview = _one_line(record.get("text"))
        elif kind == "assistant" and not record.get("tool_calls"):
            turns += 1

    if not started and not preview:
        return None  # some other .jsonl that happens to live here
    return SessionSummary(
        path=path, started=started, target=target, turns=turns, preview=preview
    )


def list_sessions(directory: Path, limit: int | None = MAX_LIST) -> list[SessionSummary]:
    """Newest first. Filenames are timestamps, so sorting by name is sorting by
    time without parsing anything.

    `limit=None` means no cap -- cleanup needs to see everything it might delete,
    whereas a listing only needs the recent ones.
    """
    if not directory.is_dir():
        return []
    try:
        files = sorted(directory.glob("*.jsonl"), key=lambda item: item.name, reverse=True)
    except OSError:
        return []

    found: list[SessionSummary] = []
    for path in files:
        if limit is not None and len(found) >= limit:
            break
        summary = summarise(path)
        if summary is not None:
            found.append(summary)
    return found


def size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def remove(path: Path) -> bool:
    """Delete one log. True when it is gone afterwards.

    A file that was already missing counts as success; a read-only USB stick
    does not. Cleanup has to report what it actually did rather than what it
    meant to do.
    """
    try:
        path.unlink()
        return True
    except OSError:
        return not path.exists()


def _tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        arguments = item.get("arguments")
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"resume_{index}"),
                name=str(item.get("name") or ""),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def _tool_results(raw: Any, pending: list[ToolCall]) -> list[ToolResult]:
    if not isinstance(raw, list):
        return []
    results: list[ToolResult] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        # Older logs carry no id, and none of them carry a name. Borrowing the
        # matching call's id is what keeps the pair consistent for the API.
        fallback = pending[index].id if index < len(pending) else f"resume_{index}"
        results.append(
            ToolResult(
                id=str(item.get("id") or fallback),
                content=str(item.get("content") or ""),
                is_error=bool(item.get("is_error")),
            )
        )
    return results


def _repair(messages: list[Message]) -> list[Message]:
    """Leave the list in a shape the next request will accept.

    A log can be cut off anywhere -- the stick was pulled, the process was
    killed -- so the tail may be an assistant turn whose tool calls were never
    answered. Anthropic rejects that outright, so the unanswered calls are
    closed off with error results rather than dropped: the model should know
    the work did not happen.
    """
    if not messages:
        return messages

    if messages[0].role != "user":
        messages.insert(0, Message("user", LOOSE_OPENING))

    last = messages[-1]
    if last.role == "assistant" and last.tool_calls:
        messages.append(
            Message(
                "user",
                tool_results=[
                    ToolResult(call.id, "工具调用未完成:会话在此中断。", is_error=True)
                    for call in last.tool_calls
                ],
            )
        )
    return messages


def load(path: Path) -> list[Message]:
    """Rebuild the neutral message list from a log."""
    messages: list[Message] = []
    pending: list[ToolCall] = []

    for record in _records(path):
        kind = record.get("type")
        if kind == "user":
            text = str(record.get("text") or "")
            if text:
                messages.append(Message("user", text))
        elif kind == "assistant":
            calls = _tool_calls(record.get("tool_calls"))
            messages.append(
                Message("assistant", str(record.get("text") or ""), tool_calls=calls)
            )
            pending = calls
        elif kind == "tool_results":
            messages.append(
                Message("user", "", tool_results=_tool_results(record.get("results"), pending))
            )
            pending = []
        elif kind == "compact":
            # A compaction replaced the conversation, so everything logged before
            # it is no longer part of the history being continued.
            messages = [
                Message("user", COMPACT_MARKER),
                Message("assistant", str(record.get("summary") or "")),
            ]
            pending = []

    return _repair(messages)
