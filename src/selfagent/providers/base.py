"""The narrow waist between provider-specific wire formats and everything else.

Anthropic's SSE events and OpenAI's chunked deltas have almost nothing in common.
Rather than let that difference leak upward into rendering, history, and the REPL,
each adapter translates its native stream into this small event union. Nothing
above this layer knows which vendor produced a token.

The same applies to tool calls. One neutral representation -- an assistant
message carrying `tool_calls`, followed by a user message carrying
`tool_results` -- is enough to reconstruct either wire format losslessly, so
sessions never store vendor-shaped content blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from ..config import ProviderSpec

Role = Literal["user", "assistant"]


@dataclass
class ToolSpec:
    """A tool the model is allowed to call. Rendered per-provider."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = field(default_factory=list)  # user only


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallsRequested:
    """The model stopped to call tools. Emitted before Done.

    Carries every call at once, including parallel ones, because the reply must
    answer all of them in a single turn.
    """

    calls: list[ToolCall]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Done:
    stop_reason: str | None = None


StreamEvent = TextDelta | ThinkingDelta | ToolCallsRequested | Usage | Done


class Provider(ABC):
    """One configured endpoint plus one model, ready to stream a completion."""

    def __init__(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str,
        *,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,
        show_thinking: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.api_key = api_key
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.show_thinking = show_thinking
        self.tools = tools or []

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[StreamEvent]:
        """Yield events until the response is complete.

        Raises ProviderError for anything the user can act on (bad key, quota,
        unreachable host), and ToolsUnsupportedError when the endpoint rejects
        the tools parameter -- the caller retries without tools in that case.
        Failures mid-stream are raised too; callers show whatever text already
        arrived before reporting.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.spec.name}/{self.model}>"
