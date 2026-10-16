import json
from pathlib import Path

import pytest

from selfagent import registry
from selfagent.config import DEFAULT_CONFIG, Config, load_config
from selfagent.errors import ConfigError


@pytest.fixture
def config(app_dir: Path) -> Config:
    (app_dir / "config.json").write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
    return load_config()


def test_no_token_uses_default(config: Config) -> None:
    target = registry.resolve(config, None)
    assert target.label == "deepseek/deepseek-v4-flash"


def test_alias(config: Config) -> None:
    assert registry.resolve(config, "claude").label == "anthropic/claude-opus-4-7"


def test_provider_slash_model(config: Config) -> None:
    assert registry.resolve(config, "anthropic/claude-sonnet-4-6").label == (
        "anthropic/claude-sonnet-4-6"
    )


def test_provider_name_uses_its_default_model(config: Config) -> None:
    assert registry.resolve(config, "deepseek").label == "deepseek/deepseek-v4-flash"


def test_other_provider_name_uses_first_listed_model(config: Config) -> None:
    assert registry.resolve(config, "anthropic").label == "anthropic/claude-opus-4-7"


def test_bare_known_model_finds_its_provider(config: Config) -> None:
    assert registry.resolve(config, "qwen-max").label == "qwen/qwen-max"


def test_unknown_token_falls_back_to_default_provider(config: Config) -> None:
    # This is what makes `-m some-brand-new-model` work against an endpoint
    # without editing config.json first.
    assert registry.resolve(config, "gpt-5-turbo").label == "deepseek/gpt-5-turbo"


def test_empty_slash_model_is_rejected(config: Config) -> None:
    with pytest.raises(ConfigError, match="缺少模型名"):
        registry.resolve(config, "anthropic/")


def test_unknown_provider_with_slash_is_rejected(config: Config) -> None:
    with pytest.raises(ConfigError, match="未知服务商"):
        registry.resolve(config, "nope/some-model")


def test_describe_mentions_every_provider(config: Config) -> None:
    text = registry.describe(config)
    for name in config.providers:
        assert name in text
