"""Turn a user-typed token into a concrete (provider, endpoint, model) triple.

`-m` has to be forgiving, because there is no way the user will remember whether
it wants a provider name, an alias, or a raw model id. Resolution order, most
specific first:

    alias  ->  "provider/model"  ->  provider name  ->  known model id
           ->  fall back to treating the token as a model on the default provider

That last fallback is what makes `-m gpt-4o` or `-m some-new-model` work against
a custom endpoint without the user editing config.json first.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, ProviderSpec
from .errors import ConfigError


@dataclass(frozen=True)
class Target:
    provider: ProviderSpec
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model}"


def resolve(config: Config, token: str | None) -> Target:
    if token is None or not token.strip():
        return Target(config.provider(config.default_provider), config.default_model)

    token = token.strip()

    if token in config.aliases:
        provider_name, model = config.aliases[token]
        return Target(config.provider(provider_name), model)

    if "/" in token:
        provider_name, _, model = token.partition("/")
        provider = config.provider(provider_name)
        if not model:
            raise ConfigError(f"'{token}' 缺少模型名,应写成 {provider_name}/<model>。")
        return Target(provider, model)

    if token in config.providers:
        provider = config.provider(token)
        # Prefer the configured default model when it belongs to this provider,
        # so `-m deepseek` honours default_model rather than guessing.
        if provider.name == config.default_provider:
            return Target(provider, config.default_model)
        if provider.models:
            return Target(provider, provider.models[0])
        raise ConfigError(
            f"服务商 '{token}' 没有配置任何模型,请写成 {token}/<model> 或先在 config.json 里补上 models。"
        )

    for provider in config.providers.values():
        if token in provider.models:
            return Target(provider, token)

    return Target(config.provider(config.default_provider), token)


def describe(config: Config) -> str:
    """A short human-readable inventory for /model with no argument."""
    lines = [f"  当前默认: {config.default_provider}/{config.default_model}", ""]
    for name, provider in config.providers.items():
        models = ", ".join(provider.models) if provider.models else "(未列模型,任意指定)"
        lines.append(f"  {name:<10} [{provider.kind}] {models}")
    if config.aliases:
        lines.append("")
        lines.append("  别名:")
        for alias, (provider_name, model) in config.aliases.items():
            lines.append(f"    {alias:<10} -> {provider_name}/{model}")
    return "\n".join(lines)
