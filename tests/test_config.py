import json
from pathlib import Path

import pytest

from selfagent.config import DEFAULT_CONFIG, load_config
from selfagent.errors import ConfigError


def write(app_dir: Path, data: dict) -> None:
    (app_dir / "config.json").write_text(json.dumps(data), encoding="utf-8")


def test_missing_config_is_materialised(app_dir: Path) -> None:
    config = load_config()
    assert (app_dir / "config.json").exists()
    assert config.default_provider == "deepseek"
    assert config.default_model == "deepseek-v4-flash"
    assert "anthropic" in config.providers


def test_default_template_is_valid(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    config = load_config()
    assert config.provider("anthropic").kind == "anthropic"
    assert config.aliases["claude"] == ("anthropic", "claude-opus-4-7")


def test_trailing_slash_is_stripped(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["providers"]["deepseek"]["base_url"] = "https://api.deepseek.com/"
    write(app_dir, data)
    assert load_config().provider("deepseek").base_url == "https://api.deepseek.com"


def test_unknown_default_provider_is_rejected(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["default_provider"] = "nope"
    write(app_dir, data)
    with pytest.raises(ConfigError, match="default_provider"):
        load_config()


def test_unknown_kind_is_rejected(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["providers"]["weird"] = {"kind": "grpc", "base_url": "http://x"}
    write(app_dir, data)
    with pytest.raises(ConfigError, match="kind"):
        load_config()


def test_missing_base_url_is_rejected(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["providers"]["bad"] = {"kind": "openai"}
    write(app_dir, data)
    with pytest.raises(ConfigError, match="base_url"):
        load_config()


def test_malformed_json_reports_line(app_dir: Path) -> None:
    (app_dir / "config.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="格式错误"):
        load_config()


def test_alias_to_missing_provider_is_dropped_not_fatal(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["aliases"]["ghost"] = {"provider": "gone", "model": "x"}
    write(app_dir, data)
    assert "ghost" not in load_config().aliases


def test_temperature_null_means_omitted(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    assert load_config().generation.temperature is None


def test_provider_lookup_error_lists_known_names(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    config = load_config()
    with pytest.raises(ConfigError, match="deepseek"):
        config.provider("nonexistent")


def test_splash_animation_defaults_on(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    assert load_config().ui.splash_animation is True


def test_splash_animation_can_be_disabled(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["splash_animation"] = False
    write(app_dir, data)
    assert load_config().ui.splash_animation is False


def test_skills_default_to_the_program_folder(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    config = load_config()
    assert config.skills.enabled is True
    assert config.skills.dir == "skills"


def test_skills_can_be_disabled(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["skills"]["enabled"] = False
    write(app_dir, data)
    assert load_config().skills.enabled is False


def test_spinner_defaults_on(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    assert load_config().ui.spinner is True


def test_spinner_can_be_disabled(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["spinner"] = False
    write(app_dir, data)
    assert load_config().ui.spinner is False
