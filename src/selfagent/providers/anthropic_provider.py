"""Anthropic adapter.

Claude needs its own adapter because the Messages API differs from Chat
Completions in ways that are not cosmetic: a top-level `system` parameter instead
of a system message, a required `max_tokens`, tool definitions using
`input_schema` rather than `parameters`, and tool results carried inside a *user*
message rather than as their own role.

The `thinking` parameter is deliberately never sent. On current models it is
adaptive by default; sending `budget_tokens` is rejected outright, and so is an
explicit disable on the most capable models. Omitting the field is the only form
that is correct across the whole family.

Tool calls are the easy part here: `get_final_message()` returns the tool_use
blocks with `input` already parsed into a dict, so none of the delta
accumulation that the OpenAI side needs applies.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from ..errors import ProviderError, ToolsUnsupportedError
from .retry import delay_for, is_transient
from .base import (
    Done,
    Message,
    Provider,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallsRequested,
    Usage,
)

_TOOL_ERROR_HINTS = ("tool", "input_schema")


def _detail(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    return str(exc)


def _looks_like_tools_error(exc: Exception) -> bool:
    return any(hint in _detail(exc).lower() for hint in _TOOL_ERROR_HINTS)


def _friendly(exc: Exception, provider_name: str, base_url: str) -> ProviderError:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ProviderError(f"[{provider_name}] 请求失败:{exc}")

    if isinstance(exc, anthropic.AuthenticationError):
        return ProviderError(
            f"[{provider_name}] API Key 无效或已过期。运行 `agent --setup` 重新录入。"
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return ProviderError(f"[{provider_name}] 该 Key 无权访问此模型。")
    if isinstance(exc, anthropic.RateLimitError):
        return ProviderError(f"[{provider_name}] 触发限流或余额不足,请稍后重试。")
    if isinstance(exc, anthropic.NotFoundError):
        return ProviderError(
            f"[{provider_name}] 模型不存在。检查模型名与 base_url({base_url})。"
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderError(
            f"[{provider_name}] 无法连接 {base_url}。检查网络、代理或 base_url 是否写错。"
        )
    if isinstance(exc, anthropic.APIStatusError):
        return ProviderError(
            f"[{provider_name}] 服务端返回 {exc.status_code}:{_detail(exc)}"
        )
    return ProviderError(f"[{provider_name}] 请求失败:{exc}")


def _serialize(messages: list[Message]) -> list[dict[str, Any]]:
    """Flatten neutral messages into Messages API shape.

    The API is strict about tool results: all `tool_result` blocks must come
    first in the user turn, that turn must immediately follow the assistant's
    `tool_use`, and every `tool_use.id` must be answered in it. The neutral model
    produces exactly that shape, so the ordering falls out for free -- but it is
    why a refused tool call still has to come back as a (error-flagged) result
    rather than being dropped.
    """
    out: list[dict[str, Any]] = []

    for message in messages:
        if message.tool_results:
            blocks: list[dict[str, Any]] = []
            for result in message.tool_results:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.content,
                }
                if result.is_error:
                    block["is_error"] = True
                blocks.append(block)
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            out.append({"role": "user", "content": blocks})

        elif message.tool_calls:
            blocks = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})

        else:
            out.append({"role": message.role, "content": message.content})

    return out


class AnthropicProvider(Provider):
    def stream(self, messages: list[Message]) -> Iterator[StreamEvent]:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(f"缺少 anthropic 依赖:{exc}") from None

        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.spec.base_url)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _serialize(messages),
        }
        if self.system:
            kwargs["system"] = self.system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.parameters,
                }
                for spec in self.tools
            ]
            # tool_choice left unset: "auto" is the default and the forced
            # variants are rejected on some models.

        # Retried here rather than around a single call, because opening the
        # stream and reading it are one `with` block. The guard is `emitted`:
        # once a delta has gone to the caller the request cannot be replayed
        # without printing the same text twice.
        total = max(1, self.retries)
        final: Any = None
        for attempt in range(1, total + 1):
            emitted = False
            try:
                with client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        if getattr(event, "type", None) != "content_block_delta":
                            continue
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            if delta.text:
                                emitted = True
                                yield TextDelta(delta.text)
                        elif delta_type == "thinking_delta" and self.show_thinking:
                            if delta.thinking:
                                emitted = True
                                yield ThinkingDelta(delta.thinking)

                    # Must run inside the context manager -- it accumulates the
                    # blocks that arrived over the stream, and the object is
                    # unusable once the stream is closed. Its tool_use inputs are
                    # already parsed, which is why no input_json_delta handling
                    # appears here.
                    final = stream.get_final_message()
            except Exception as exc:
                if not emitted and attempt < total and is_transient(exc):
                    delay = delay_for(attempt)
                    if self.on_retry is not None:
                        self.on_retry(attempt, total, delay, exc)
                    time.sleep(delay)
                    continue

                try:
                    import anthropic
                except ImportError:  # pragma: no cover
                    raise
                if (
                    isinstance(exc, anthropic.BadRequestError)
                    and self.tools
                    and _looks_like_tools_error(exc)
                ):
                    raise ToolsUnsupportedError(
                        f"[{self.spec.name}] 该端点不支持工具调用:{_detail(exc)}"
                    ) from None
                raise _friendly(exc, self.spec.name, self.spec.base_url) from None
            break

        calls = _extract_tool_calls(final)
        if calls:
            yield ToolCallsRequested(calls)

        usage = getattr(final, "usage", None)
        if usage is not None:
            yield Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        else:
            yield Usage()

        yield Done(getattr(final, "stop_reason", None))


def _extract_tool_calls(final: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for block in getattr(final, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        arguments = getattr(block, "input", None)
        calls.append(
            ToolCall(
                id=getattr(block, "id", "") or "",
                name=getattr(block, "name", "") or "",
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls
