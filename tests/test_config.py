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


def test_prompt_box_defaults_on(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    assert load_config().ui.prompt_box is True


def test_prompt_box_can_be_disabled(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["prompt_box"] = False
    write(app_dir, data)
    assert load_config().ui.prompt_box is False


# ---- theme and identity ----------------------------------------------


def test_theme_defaults_to_the_original_gradient(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    config = load_config()
    assert config.ui.theme_top == (0x22, 0xD3, 0xEE)
    assert config.ui.theme_bottom == (0xA8, 0x55, 0xF7)
    assert config.ui.theme_accent == config.ui.theme_top


def test_theme_colours_can_be_overridden(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["theme"] = {
        "top": "#ff6b6b",
        "bottom": "f59e0b",  # the leading # is optional
        "accent": "#abc",    # three-digit shorthand
    }
    write(app_dir, data)

    config = load_config()
    assert config.ui.theme_top == (0xFF, 0x6B, 0x6B)
    assert config.ui.theme_bottom == (0xF5, 0x9E, 0x0B)
    assert config.ui.theme_accent == (0xAA, 0xBB, 0xCC)


def test_accent_defaults_to_the_configured_top(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["theme"] = {"top": "#123456", "bottom": "#654321"}
    write(app_dir, data)
    config = load_config()
    assert config.ui.theme_accent == (0x12, 0x34, 0x56)


def test_a_malformed_theme_colour_names_the_field(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["theme"] = {"top": "not-a-colour"}
    write(app_dir, data)
    with pytest.raises(ConfigError, match="ui.theme.top"):
        load_config()


def test_theme_must_be_an_object(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["ui"]["theme"] = "cyan-violet"
    write(app_dir, data)
    with pytest.raises(ConfigError, match="ui.theme"):
        load_config()


def test_agent_defaults_leave_the_system_prompt_untouched(app_dir: Path) -> None:
    write(app_dir, DEFAULT_CONFIG)
    agent = load_config().agent
    assert agent.name == "SelfAgent"
    assert agent.persona == ""
    assert agent.instructions == ""
    assert agent.system_block() == ""


def test_agent_identity_is_parsed(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["agent"] = {
        "name": "  小助手  ",
        "persona": "回答先给结论",
        "instructions": "引用文件时给出路径",
    }
    write(app_dir, data)

    agent = load_config().agent
    assert agent.name == "小助手"
    block = agent.system_block()
    assert "Your name is 小助手" in block
    assert "Role: 回答先给结论" in block
    assert "引用文件时给出路径" in block


def test_an_empty_agent_name_falls_back_to_the_product_name(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["agent"]["name"] = "   "
    write(app_dir, data)
    assert load_config().agent.name == "SelfAgent"


def test_agent_must_be_an_object(app_dir: Path) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["agent"] = "小助手"
    write(app_dir, data)
    with pytest.raises(ConfigError, match="agent"):
        load_config()


# ---- writing the config back (the /theme, /name, ... commands) --------


def test_parse_color_accepts_the_forms_the_command_documents() -> None:
    from selfagent.config import parse_color

    assert parse_color("#FF0000") == (255, 0, 0)
    assert parse_color("00ff00") == (0, 255, 0)
    assert parse_color("#0f0") == (0, 255, 0)

    with pytest.raises(ConfigError, match="theme.accent"):
        parse_color("nope", "theme.accent")


def test_saving_round_trips_and_leaves_no_temp_file(app_dir: Path) -> None:
    config = load_config()
    config.raw["agent"]["name"] = "小助手"
    config.save()

    saved = json.loads((app_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["agent"]["name"] == "小助手"
    assert not (app_dir / "config.json.tmp").exists()

    # And the file we just wrote is still a valid config.
    assert load_config().agent.name == "小助手"
