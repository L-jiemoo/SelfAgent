"""Glue: one turn of conversation, end to end.

Both the REPL and one-shot paths drive the same object, so a bug in streaming
handling cannot differ between the two modes.

A "turn" is no longer a single request. When tools are enabled the model can
stop, ask for a file to be read, see the result, and ask again -- so `ask()`
runs a bounded loop. The bound matters: nothing stops a model from re-calling
the same tool forever, and each round costs a request.
"""

from __future__ import annotations

from typing import Iterator

from .config import Config
from .errors import SelfAgentError, ToolsUnsupportedError
from .providers.base import (
    Message,  # noqa: F401 - re-exported for callers/tests
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallsRequested,
    ToolResult,
    ToolSpec,
    Usage,
)
from .providers.factory import build_provider, require_key
from .registry import Target, resolve
from .render import Renderer
from .secrets import SecretStore
from .session import Session
from .tools import WriteRecord

# A transcript longer than this is truncated from the front before being sent to
# the summariser: the tail is what matters and the head is what the model is
# about to forget anyway.
MAX_TRANSCRIPT = 60_000

_COMPACT_PROMPT = (
    "Summarise the conversation below so it can replace the original without "
    "losing anything that matters: decisions taken, file paths, code, and open "
    "questions. Drop pleasantries and repetition. Reply with the summary only, "
    "in the language the conversation used.\n\n---\n{transcript}\n---"
)


class Engine:
    def __init__(
        self,
        config: Config,
        secrets: SecretStore,
        renderer: Renderer,
        *,
        model_token: str | None = None,
        persist: bool = True,
        toolbox=None,
        skills=None,
        context_path=None,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self.renderer = renderer
        self.toolbox = toolbox
        # Held only so the REPL can list them; the model learns about skills
        # through the system prompt, not through this attribute.
        self.skills = list(skills or [])
        # Same: shown by /status so "why is it ignoring my conventions" has an
        # answer that is not guesswork.
        self.context_path = context_path
        self.target: Target = resolve(config, model_token)
        self.session = Session(
            config,
            self.target.label,
            persist=persist,
            on_warning=renderer.warning,
        )
        # Flipped off for the rest of the session once an endpoint proves it
        # cannot take the tools parameter.
        self.tools_enabled = bool(config.tools.enabled and toolbox is not None)
        # Kept so /retry can send the same question again without the user
        # retyping it -- the whole reason to have the command.
        self.last_question: str | None = None

    def switch(self, token: str | None) -> Target:
        """Change model. Existing history is kept -- switching mid-conversation
        is a legitimate way to compare answers, and clearing it would be a
        surprising side effect of what looks like a display setting."""
        self.target = resolve(self.config, token)
        self.session.label = self.target.label
        # Tool support is a property of the endpoint, not of the session, so a
        # switch deserves a fresh attempt.
        self.tools_enabled = bool(self.config.tools.enabled and self.toolbox is not None)
        return self.target

    def set_tools_mode(self, mode: str) -> str:
        """Switch the file tools between off, read-only and read-write.

        Read-only is the interesting one: it is the mode you want on a machine
        that is not yours when you still want the model to be able to look
        around. Only ever narrows or restores what was configured at launch --
        with no toolbox there is nothing to switch on.
        """
        if mode == "off":
            self.tools_enabled = False
            return "文件工具已关闭。"
        if self.toolbox is None:
            raise SelfAgentError(
                "本次启动没有文件工具(可能用了 --no-tools,或工作目录不存在),无法打开。"
            )
        self.tools_enabled = self.config.tools.enabled
        self.toolbox.allow_write = mode == "on"
        if mode == "on":
            return "文件工具:可读可写,每次写入前确认。"
        return "文件工具:只读(模型能看,但改不了)。"

    # ---- taking it back -------------------------------------------------

    def undo(self) -> str | None:
        """Roll back the most recent file write. None when there is none left.

        The record holds the file's previous contents, captured by the toolbox
        at write time -- the confirmation callback already had them, and they
        were being thrown away.
        """
        if self.toolbox is None:
            raise SelfAgentError("本次启动没有文件工具,没有可撤销的修改。")
        return self.toolbox.undo_last()

    def audit(self) -> list[WriteRecord]:
        """Every file this session has written, newest first."""
        if self.toolbox is None:
            return []
        return list(reversed(self.toolbox.writes))

    # ---- compaction ---------------------------------------------------
    def compact(self) -> str:
        """Replace the conversation with a summary of itself.

        Deliberately not routed through ask(): the point is to *replace* the
        history, and ask() would append to it. The original is restored on any
        failure, because a compaction that loses the conversation is far worse
        than one that simply does not happen.
        """
        if not self.session.messages:
            raise SelfAgentError("当前没有可压缩的对话。")

        key = require_key(self.secrets, self.target)
        transcript = self._transcript()

        saved_messages = list(self.session.messages)
        saved_tools = self.tools_enabled
        # A summary request that could call tools would be a request that edits
        # a file instead of summarising.
        self.tools_enabled = False
        try:
            self.session.messages = [
                Message("user", _COMPACT_PROMPT.format(transcript=transcript))
            ]
            self.renderer.thinking_start()
            parts: list[str] = []
            for event in self._stream(key):
                self.renderer.handle(event)
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                elif isinstance(event, Usage):
                    self.session.add_usage(event)
            summary = "".join(parts).strip()
        except BaseException:
            self.session.messages = saved_messages
            raise
        finally:
            self.tools_enabled = saved_tools
            self.renderer.finish()

        if not summary:
            self.session.messages = saved_messages
            raise SelfAgentError("模型没有返回摘要,原对话已保留。")

        self.session.replace_with_summary(summary)
        return summary

    def _transcript(self) -> str:
        lines: list[str] = []
        for message in self.session.messages:
            if message.tool_results:
                for result in message.tool_results:
                    lines.append(f"[tool result] {result.content[:500]}")
            elif message.tool_calls:
                names = ", ".join(call.name for call in message.tool_calls)
                lines.append(f"[assistant called: {names}] {message.content}")
            else:
                lines.append(f"[{message.role}] {message.content}")

        transcript = "\n".join(lines)
        if len(transcript) > MAX_TRANSCRIPT:
            transcript = "(更早的部分已省略)\n" + transcript[-MAX_TRANSCRIPT:]
        return transcript

    # ---- one turn -----------------------------------------------------

    def ask(self, text: str) -> str:
        # Resolve credentials before touching history, so a missing key leaves
        # the conversation exactly as it was.
        key = require_key(self.secrets, self.target)
        self.last_question = text
        self.session.add_user(text)

        last_text = ""
        try:
            for _ in range(max(1, self.config.tools.max_steps)):
                text_parts: list[str] = []
                calls: list[ToolCall] = []

                # Started before the stream is consumed: the wait for the first
                # token is the whole reason the marker exists.
                self.renderer.thinking_start()

                for event in self._stream(key):
                    self.renderer.handle(event)
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                    elif isinstance(event, ToolCallsRequested):
                        calls = event.calls
                    elif isinstance(event, Usage):
                        self.session.add_usage(event)

                self.renderer.finish()
                last_text = "".join(text_parts)

                if not calls:
                    self.session.add_assistant(last_text)
                    return last_text

                self.session.add_assistant(last_text, tool_calls=calls)
                self.session.add_tool_results(self._run_tools(calls))
            else:
                self.renderer.warning(
                    f"已达到单轮工具调用上限({self.config.tools.max_steps} 次),停止继续调用。"
                )
                self.session.add_assistant(last_text)
                return last_text
        except BaseException:
            self.renderer.finish()
            self.session.repair_tail()
            raise

    # ---- helpers ------------------------------------------------------

    def _tool_specs(self) -> list[ToolSpec]:
        if not self.tools_enabled or self.toolbox is None:
            return []
        if not self.target.provider.tools:
            return []
        return self.toolbox.specs()

    def _stream(self, key: str) -> Iterator[StreamEvent]:
        """Stream one request, degrading to plain chat if tools are refused.

        Support for the `tools` parameter is inconsistent across
        OpenAI-compatible endpoints, and sending it can break a request that
        would otherwise have worked. Rather than ship a whitelist that rots,
        try with tools and fall back once, then remember for the session.
        """
        tools = self._tool_specs()
        provider = build_provider(
            self.target, key, self.config, tools=tools, on_retry=self._notify_retry
        )
        try:
            yield from provider.stream(self.session.context())
        except ToolsUnsupportedError as exc:
            if not tools:
                raise
            self.tools_enabled = False
            self.renderer.warning(f"{exc} 本次会话改用纯文本模式。")
            fallback = build_provider(
                self.target, key, self.config, tools=[], on_retry=self._notify_retry
            )
            yield from fallback.stream(self.session.context())

    def _notify_retry(self, attempt: int, total: int, delay: float, exc: Exception) -> None:
        """Say so before sleeping: a silent pause reads as a hang."""
        self.renderer.warning(
            f"请求失败({exc.__class__.__name__}),{delay:.0f} 秒后重试"
            f"({attempt + 1}/{total})。"
        )
        # warning() stops the spinner; put it back, since the wait is not over.
        self.renderer.thinking_start()

    def retry(self) -> str:
        """Send the most recent question again.

        Useful when a request died part-way: the failed turn is repaired out of
        the history first, so this does not leave a dangling question behind.
        """
        if not self.last_question:
            raise SelfAgentError("还没有问过任何问题,没有可重发的。")
        return self.ask(self.last_question)

    def _run_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            self.renderer.tool_start(call)
            if self.toolbox is None:
                result = ToolResult(call.id, "工具未启用。", is_error=True)
            else:
                result = self.toolbox.execute(call)
            self.renderer.tool_end(result)
            results.append(result)
        return results
