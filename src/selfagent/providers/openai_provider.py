"""One adapter for eight providers.

DeepSeek, OpenAI, Qwen, Kimi, GLM, Doubao and any self-hosted endpoint all speak
the same Chat Completions wire format. The only thing that distinguishes them is
base_url and the model string, and both of those are already in config.json. So
this is the whole adapter layer for all of them.

`chat.completions` is used exclusively and never `responses.create`: the
compatibility shims the Chinese providers run generally implement Chat
Completions only, and would 404 on the newer surface.

Tool calls are the fiddly part here. Unlike Anthropic, there is no assembled
"final message" to read them from, so deltas have to be accumulated by hand --
and the providers disagree about how to label the fragments. See
`_accumulate_tool_calls`.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from ..errors import ProviderError, ToolsUnsupportedError
from .retry import open_with_retries
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

# Substrings that identify a 400 as being about tools rather than something the
# user should see. There is no standard error code for this, so it is a text
# match -- the alternative is a hard-coded provider allowlist that rots.
_TOOL_ERROR_HINTS = ("tool_choice", "tools", "tool_calls", "tool call", "function calling")


def _looks_like_tools_error(exc: Exception) -> bool:
    text = f"{_detail(exc)} {exc}".lower()
    return any(hint in text for hint in _TOOL_ERROR_HINTS)


def _friendly(exc: Exception, provider_name: str, base_url: str) -> ProviderError:
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return ProviderError(f"[{provider_name}] 请求失败:{exc}")

    if isinstance(exc, openai.AuthenticationError):
        return ProviderError(
            f"[{provider_name}] API Key 无效或已过期。运行 `agent --setup` 重新录入。"
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return ProviderError(f"[{provider_name}] 该 Key 无权访问此模型。")
    if isinstance(exc, openai.RateLimitError):
        return ProviderError(f"[{provider_name}] 触发限流或余额不足,请稍后重试。")
    if isinstance(exc, openai.NotFoundError):
        return ProviderError(
            f"[{provider_name}] 接口或模型不存在。检查 config.json 里的 base_url({base_url})与模型名。"
        )
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(
            f"[{provider_name}] 无法连接 {base_url}。检查网络、代理或 base_url 是否写错。"
        )
    if isinstance(exc, openai.APIStatusError):
        return ProviderError(f"[{provider_name}] 服务端返回 {exc.status_code}:{_detail(exc)}")
    return ProviderError(f"[{provider_name}] 请求失败:{exc}")


def _detail(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(body.get("message"), str):
            return body["message"]
    return str(exc)


def _serialize(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """Flatten neutral messages into Chat Completions shape.

    The one real translation: this API wants tool results as their own
    `role: "tool"` messages, whereas the neutral model keeps them attached to
    the user turn that follows the assistant's call.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        if message.tool_results:
            for result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.id,
                        "content": result.content,
                    }
                )
            if message.content:
                out.append({"role": "user", "content": message.content})

        elif message.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    # Must be None rather than "" -- some endpoints reject an
                    # empty string alongside tool_calls.
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )

        else:
            out.append({"role": message.role, "content": message.content})

    return out


def _accumulate_tool_calls(buffer: dict[int, dict[str, str]], fragments: Any) -> None:
    """Merge one chunk's tool_call fragments into the running buffer.

    The spec says fragments are keyed by `index`, but providers deviate in two
    documented ways, both of which produce silently corrupted arguments:

    * Zhipu GLM emits several calls in one delta with no `index` at all, so a
      naive positional fallback folds them into slot 0 and concatenates their
      argument strings into invalid JSON.
    * openai-python had a bug where distinct calls shared an index.

    So: prefer `index`, fall back to position, and when a fragment *starts a new
    named call* in a slot that is already named, treat it as a new call.
    """
    for position, fragment in enumerate(fragments or []):
        index = getattr(fragment, "index", None)
        function = getattr(fragment, "function", None)
        name = getattr(function, "name", None) if function else None

        if index is None:
            index = position
            if name and index in buffer and buffer[index]["name"]:
                index = max(buffer) + 1

        slot = buffer.setdefault(index, {"id": "", "name": "", "args": ""})
        if getattr(fragment, "id", None):
            slot["id"] = fragment.id
        if name:
            slot["name"] = name
        arguments = getattr(function, "arguments", None) if function else None
        if arguments:
            slot["args"] += arguments


def _finish_tool_calls(buffer: dict[int, dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(buffer):
        slot = buffer[index]
        if not slot["name"]:
            continue
        try:
            arguments = json.loads(slot["args"]) if slot["args"].strip() else {}
        except json.JSONDecodeError:
            # Malformed arguments are the model's problem to fix, and it can
            # only fix them if it is told. An empty dict surfaces as a missing
            # required argument, which points at the real cause.
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"], arguments=arguments)
        )
    return calls


class OpenAICompatProvider(Provider):
    def stream(self, messages: list[Message]) -> Iterator[StreamEvent]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(f"缺少 openai 依赖:{exc}") from None

        client = OpenAI(api_key=self.api_key, base_url=self.spec.base_url)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _serialize(messages, self.system),
            "stream": True,
        }
        # Omitting temperature matters: thinking-capable models reject sampling
        # params outright, so null in config means "don't send the field".
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in self.tools
            ]
            # tool_choice is deliberately left unset. "auto" is the default, and
            # DeepSeek and Qwen reject "required"/forced choices outright when
            # thinking is on.

        # Only the opening is retried: the loop below may already have handed
        # text to the caller, and replaying it would print it twice.
        chunk_iter = open_with_retries(
            lambda: self._open_stream(client, kwargs), self.retries, self.on_retry
        )

        saw_usage = False
        stop_reason: str | None = None
        buffer: dict[int, dict[str, str]] = {}

        try:
            for chunk in chunk_iter:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    saw_usage = True
                    yield Usage(
                        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    )

                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    stop_reason = choice.finish_reason

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                # DeepSeek-style reasoning models expose a separate field. Some
                # providers call it reasoning_content, others reasoning.
                if self.show_thinking:
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if reasoning:
                        yield ThinkingDelta(reasoning)

                content = getattr(delta, "content", None)
                if content:
                    yield TextDelta(content)

                _accumulate_tool_calls(buffer, getattr(delta, "tool_calls", None))
        except Exception as exc:
            raise _friendly(exc, self.spec.name, self.spec.base_url) from None

        if buffer:
            calls = _finish_tool_calls(buffer)
            if calls:
                yield ToolCallsRequested(calls)

        if not saw_usage:
            yield Usage()
        yield Done(stop_reason)

    def _open_stream(self, client: Any, kwargs: dict[str, Any]):
        """Open the stream, degrading in two stages.

        Order matters. A 400 about tools must not be mistaken for the
        stream_options case, because stripping stream_options would not fix it
        and the retry would fail confusingly.
        """
        try:
            return client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except Exception as exc:
            try:
                import openai
            except ImportError:  # pragma: no cover
                raise

            if isinstance(exc, openai.BadRequestError):
                if kwargs.get("tools") and _looks_like_tools_error(exc):
                    raise ToolsUnsupportedError(
                        f"[{self.spec.name}] 该端点不支持工具调用:{_detail(exc)}"
                    ) from None
                # Most endpoints accept stream_options; some shims 400 on it.
                without = {k: v for k, v in kwargs.items() if k != "stream_options"}
                try:
                    return client.chat.completions.create(**without)
                except Exception as retry_exc:
                    raise _friendly(
                        retry_exc, self.spec.name, self.spec.base_url
                    ) from None
            raise _friendly(exc, self.spec.name, self.spec.base_url) from None
