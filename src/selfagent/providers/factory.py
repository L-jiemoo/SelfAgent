"""Pick an adapter from the provider's `kind` field."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError, ProviderError
from ..registry import Target
from .anthropic_provider import AnthropicProvider
from .base import Provider, ToolSpec
from .openai_provider import OpenAICompatProvider

_ADAPTERS = {
    "openai": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(
    target: Target,
    api_key: str,
    config: Config,
    *,
    tools: list[ToolSpec] | None = None,
    on_retry=None,
) -> Provider:
    adapter = _ADAPTERS.get(target.provider.kind)
    if adapter is None:
        raise ConfigError(
            f"服务商 '{target.provider.name}' 的 kind '{target.provider.kind}' 没有对应适配器。"
        )

    return adapter(
        spec=target.provider,
        model=target.model,
        api_key=api_key,
        system=config.generation.system,
        max_tokens=config.generation.max_tokens,
        temperature=config.generation.temperature,
        show_thinking=config.ui.show_thinking,
        tools=tools,
        retries=config.generation.retries,
        on_retry=on_retry,
    )


def require_key(secrets, target: Target) -> str:
    key = secrets.api_key(target.provider.name)
    if not key:
        raise ProviderError(
            f"没有为 '{target.provider.name}' 配置 API Key。"
            f"运行 `agent --setup` 录入,或在交互模式里用 /key {target.provider.name} <key>。"
        )
    return key
