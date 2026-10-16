"""Conversation state and its on-disk log.

Persistence is a convenience, never a dependency. The app runs from removable
media that may be read-only, full, or yanked out mid-session. Every write here is
best-effort: the first failure prints one warning, sets a flag, and from then on
the session quietly continues in memory. A logging problem must never be able to
end a conversation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import paths
from .config import Config
from .providers.base import Message, ToolCall, ToolResult, Usage

# Stands in for everything the summary replaces. It exists so the collapsed
# history still opens with a user turn, which Anthropic requires outright.
COMPACT_MARKER = "(此前对话已压缩为下面的摘要)"


class Session:
    def __init__(
        self,
        config: Config,
        label: str,
        *,
        persist: bool = True,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.label = label
        self.messages: list[Message] = []
        self.usage = Usage()
        self.turns = 0

        self._on_warning = on_warning or (lambda _msg: None)
        self._history_file: Path | None = None
        self._history_disabled = False
        self._warned = False

        if persist:
            self._open_history()

    # ---- conversation -------------------------------------------------

    def add_user(self, text: str) -> None:
        self.messages.append(Message("user", text))
        self._record({"type": "user", "text": text})

    def add_assistant(self, text: str, tool_calls: list[ToolCall] | None = None) -> None:
        self.messages.append(Message("assistant", text, tool_calls=tool_calls or []))
        if not tool_calls:
            # Only a reply that isn't asking for tools counts as a finished turn;
            # otherwise the counter would tick once per tool round-trip.
            self.turns += 1
        record: dict = {"type": "assistant", "text": text}
        if tool_calls:
            # The id is what lets a resumed session pair a call with its result.
            # Older logs predate this field; history.load rebuilds it by
            # position when it is missing.
            record["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in tool_calls
            ]
        self._record(record)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append(Message("user", "", tool_results=results))
        self._record(
            {
                "type": "tool_results",
                "results": [
                    {"id": r.id, "is_error": r.is_error, "content": r.content[:2000]}
                    for r in results
                ],
            }
        )

    def repair_tail(self) -> None:
        """Leave the conversation in a shape the next request will accept.

        A failure can strand history in three ways, and two of them are rejected
        outright by Anthropic: a dangling plain user turn, a tool_use with no
        answer, or tool results with no assistant turn after them.
        """
        if not self.messages:
            return
        last = self.messages[-1]

        if last.tool_results:
            self.messages.append(Message("assistant", "(上一轮未完成)"))
        elif last.role == "assistant" and last.tool_calls:
            self.messages.append(
                Message(
                    "user",
                    tool_results=[
                        ToolResult(call.id, "工具调用未完成:请求中断。", is_error=True)
                        for call in last.tool_calls
                    ],
                )
            )
        elif last.role == "user":
            self.messages.pop()

    def add_usage(self, usage: Usage) -> None:
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens

    def clear(self) -> None:
        self.messages.clear()
        self.turns = 0

    def replace_with_summary(self, summary: str) -> None:
        """Collapse the conversation into a single exchange.

        Two messages rather than one: both APIs require the history to alternate
        roles, and Anthropic additionally requires it to open with a user turn.
        The summary therefore rides as the assistant's reply to a synthetic
        prompt, and the next real question follows it naturally.
        """
        self.messages.clear()
        self.messages.append(Message("user", COMPACT_MARKER))
        self.messages.append(Message("assistant", summary))
        self._record({"type": "compact", "summary": summary[:2000]})

    @property
    def history_path(self) -> Path | None:
        return self._history_file

    def restore(self, messages: list[Message]) -> None:
        """Adopt a conversation rebuilt from an earlier session's log.

        The log is the source of truth for what was said; the turn counter is
        recomputed rather than carried over, because the log it came from may
        have been cut short.
        """
        self.messages = list(messages)
        self.turns = sum(
            1
            for message in self.messages
            if message.role == "assistant" and not message.tool_calls
        )
        self._record({"type": "resume", "messages": len(self.messages)})

    def pop_last_user(self) -> None:
        """Undo a user turn whose request never produced any output.

        Leaving it in place would put two user messages back to back, which
        Anthropic rejects outright -- so a failed turn must not poison the next
        one.
        """
        if self.messages and self.messages[-1].role == "user":
            self.messages.pop()

    def context(self) -> list[Message]:
        """The slice of history actually sent to the API.

        `max_turns` trims the request, not the log.

        Trimming has to land on turn boundaries rather than on a raw message
        count. One user question can now span several messages -- assistant
        tool_use, the tool results, the assistant's follow-up -- and cutting
        between them hands the API an orphaned tool_use that it rejects
        outright. A turn starts at a user message that is *not* carrying tool
        results, so cutting there always yields a valid conversation.
        """
        limit = self.config.history.max_turns
        if limit <= 0 or not self.messages:
            return list(self.messages)

        starts = [
            index
            for index, message in enumerate(self.messages)
            if message.role == "user" and not message.tool_results
        ]
        if len(starts) <= limit:
            return list(self.messages)
        return self.messages[starts[-limit] :]

    # ---- persistence --------------------------------------------------

    def _open_history(self) -> None:
        if not self.config.history.enabled:
            return
        try:
            directory = paths.resolve_data_dir(self.config.history.dir)
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._history_file = directory / f"{stamp}.jsonl"
            self._write_line(
                {
                    "type": "session_start",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "target": self.label,
                }
            )
        except OSError as exc:
            self._disable_history(f"历史目录不可用({exc.strerror or exc})")

    def _record(self, payload: dict) -> None:
        payload["time"] = datetime.now().isoformat(timespec="seconds")
        self._write_line(payload)

    def _write_line(self, payload: dict) -> None:
        if self._history_file is None or self._history_disabled:
            return
        try:
            with self._history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._disable_history(f"历史无法保存({exc.strerror or exc})")

    def _disable_history(self, reason: str) -> None:
        self._history_disabled = True
        if not self._warned:
            self._warned = True
            self._on_warning(f"{reason},本次会话改为仅内存,对话不受影响。")
