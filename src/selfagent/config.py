"""Plaintext, hand-editable settings.

Everything in config.json is non-secret on purpose. Provider endpoints and model
names drift constantly, and the user must be able to fix them in Notepad on a
machine we've never seen -- without a passphrase and without a rebuild. Only API
keys live in the encrypted store (see secrets.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .errors import ConfigError

VALID_KINDS = ("openai", "anthropic")

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "default_provider": "deepseek",
    "default_model": "deepseek-v4-flash",
    "providers": {
        "deepseek": {
            "kind": "openai",
            "base_url": "https://api.deepseek.com",
            "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "tools": True,
        },
        "openai": {
            "kind": "openai",
            "base_url": "https://api.openai.com/v1",
            "models": [],
            "tools": True,
        },
        "anthropic": {
            "kind": "anthropic",
            "base_url": "https://api.anthropic.com",
            "models": [
                "claude-opus-4-7",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
            "tools": True,
        },
        "qwen": {
            "kind": "openai",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": ["qwen-max", "qwen-plus"],
            "tools": True,
        },
        "kimi": {
            "kind": "openai",
            "base_url": "https://api.moonshot.cn/v1",
            "models": ["kimi-k2-0905-preview"],
            "tools": False,
        },
        "zhipu": {
            "kind": "openai",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "models": ["glm-4-plus"],
            "tools": False,
        },
        "doubao": {
            "kind": "openai",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "models": [],
            "tools": True,
        },
        "custom": {
            "kind": "openai",
            "base_url": "http://localhost:11434/v1",
            "models": [],
            "tools": True,
        },
    },
    "aliases": {
        "claude": {"provider": "anthropic", "model": "claude-opus-4-7"},
        "flash": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "pro": {"provider": "deepseek", "model": "deepseek-v4-pro"},
    },
    "generation": {
        "max_tokens": 8192,
        "temperature": None,
        "system": "You are a helpful assistant.",
    },
    "ui": {
        "stream": True,
        "color": True,
        "show_thinking": False,
        "splash_animation": True,
        "spinner": True,
    },
    "history": {"enabled": True, "dir": "history", "max_turns": 50},
    "skills": {"enabled": True, "dir": "skills"},
    "tools": {
        "enabled": True,
        "workspace": None,
        "confirm_writes": True,
        "max_steps": 10,
        "max_read_bytes": 262144,
    },
}


@dataclass
class ProviderSpec:
    name: str
    kind: str
    base_url: str
    models: list[str] = field(default_factory=list)
    # Function-calling support varies wildly across OpenAI-compatible endpoints,
    # and sending `tools` can break a request that otherwise works. This is the
    # manual override; the adapters also degrade automatically on a tools-
    # related 400, so a wrong `True` here is recoverable at runtime.
    tools: bool = True


@dataclass
class Generation:
    max_tokens: int = 8192
    temperature: float | None = None
    system: str | None = None


@dataclass
class UIConfig:
    stream: bool = True
    color: bool = True
    show_thinking: bool = False
    # Animated startup logo. Purely cosmetic, and off is one line in
    # config.json -- which matters on the slow USB sticks this tool is built
    # for, and on consoles where escape codes are not welcome.
    splash_animation: bool = True
    # The "still working" marker shown while waiting for the first token.
    spinner: bool = True


@dataclass
class HistoryConfig:
    enabled: bool = True
    dir: str = "history"
    max_turns: int = 50


@dataclass
class SkillsConfig:
    enabled: bool = True
    # Relative to the program folder, so skills ride along on the USB stick.
    # Absolute paths are honoured too, for a shared library on a local disk.
    dir: str = "skills"


@dataclass
class ToolsConfig:
    enabled: bool = True
    # None means "the directory the user launched from" -- see tools.auto_root.
    workspace: str | None = None
    confirm_writes: bool = True
    # Caps the tool-call loop. Without it a model that keeps re-calling the same
    # tool spins forever.
    max_steps: int = 10
    max_read_bytes: int = 262144


@dataclass
class Config:
    path: Path
    raw: dict[str, Any]
    providers: dict[str, ProviderSpec]
    aliases: dict[str, tuple[str, str]]
    default_provider: str
    default_model: str
    generation: Generation
    ui: UIConfig
    history: HistoryConfig
    skills: SkillsConfig
    tools: ToolsConfig

    def provider(self, name: str) -> ProviderSpec:
        try:
            return self.providers[name]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise ConfigError(
                f"未知服务商 '{name}'。config.json 中已配置:{known}"
            ) from None

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _as_bool(value: Any, default: bool) -> bool:
    return bool(value) if isinstance(value, bool) else default


def _as_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _parse_providers(raw: Any) -> dict[str, ProviderSpec]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("config.json 的 providers 必须是一个非空对象。")

    out: dict[str, ProviderSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"服务商 '{name}' 的配置必须是对象。")

        kind = spec.get("kind", "openai")
        if kind not in VALID_KINDS:
            raise ConfigError(
                f"服务商 '{name}' 的 kind 为 '{kind}',只支持:{', '.join(VALID_KINDS)}"
            )

        base_url = spec.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigError(f"服务商 '{name}' 缺少 base_url。")

        models = spec.get("models", [])
        if not isinstance(models, list):
            models = []

        out[name] = ProviderSpec(
            name=name,
            kind=kind,
            base_url=base_url.rstrip("/"),
            models=[m for m in models if isinstance(m, str)],
            tools=_as_bool(spec.get("tools"), True),
        )
    return out


def _parse_aliases(raw: Any, providers: dict[str, ProviderSpec]) -> dict[str, tuple[str, str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, str]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        provider = spec.get("provider")
        model = spec.get("model")
        if isinstance(provider, str) and isinstance(model, str):
            # Drop aliases pointing at providers that no longer exist rather
            # than failing the whole load over a stale entry.
            if provider in providers:
                out[name] = (provider, model)
    return out


def load_config() -> Config:
    path = paths.config_path()

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raw = json.loads(json.dumps(DEFAULT_CONFIG))
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"config.json 格式错误(第 {exc.lineno} 行):{exc.msg}"
            ) from None
        if not isinstance(raw, dict):
            raise ConfigError("config.json 的顶层必须是一个对象。")

    providers = _parse_providers(raw.get("providers"))
    aliases = _parse_aliases(raw.get("aliases"), providers)

    default_provider = raw.get("default_provider") or next(iter(providers))
    if default_provider not in providers:
        raise ConfigError(f"default_provider '{default_provider}' 不在 providers 中。")

    default_model = raw.get("default_model")
    if not isinstance(default_model, str) or not default_model.strip():
        spec = providers[default_provider]
        if not spec.models:
            raise ConfigError(
                f"default_model 未设置,且服务商 '{default_provider}' 没有候选模型。"
            )
        default_model = spec.models[0]

    gen_raw = raw.get("generation") or {}
    temperature = gen_raw.get("temperature")
    gen = Generation(
        max_tokens=_as_int(gen_raw.get("max_tokens"), 8192),
        temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
        system=gen_raw.get("system") if isinstance(gen_raw.get("system"), str) else None,
    )

    ui_raw = raw.get("ui") or {}
    ui = UIConfig(
        stream=_as_bool(ui_raw.get("stream"), True),
        color=_as_bool(ui_raw.get("color"), True),
        show_thinking=_as_bool(ui_raw.get("show_thinking"), False),
        splash_animation=_as_bool(ui_raw.get("splash_animation"), True),
        spinner=_as_bool(ui_raw.get("spinner"), True),
    )

    hist_raw = raw.get("history") or {}
    history = HistoryConfig(
        enabled=_as_bool(hist_raw.get("enabled"), True),
        dir=hist_raw.get("dir") if isinstance(hist_raw.get("dir"), str) else "history",
        max_turns=max(0, _as_int(hist_raw.get("max_turns"), 50)),
    )

    skills_raw = raw.get("skills") or {}
    skills = SkillsConfig(
        enabled=_as_bool(skills_raw.get("enabled"), True),
        dir=skills_raw.get("dir") if isinstance(skills_raw.get("dir"), str) else "skills",
    )

    tools_raw = raw.get("tools") or {}
    workspace = tools_raw.get("workspace")
    tools = ToolsConfig(
        enabled=_as_bool(tools_raw.get("enabled"), True),
        workspace=workspace if isinstance(workspace, str) and workspace.strip() else None,
        confirm_writes=_as_bool(tools_raw.get("confirm_writes"), True),
        max_steps=max(1, _as_int(tools_raw.get("max_steps"), 10)),
        max_read_bytes=max(1024, _as_int(tools_raw.get("max_read_bytes"), 262144)),
    )

    return Config(
        path=path,
        raw=raw,
        providers=providers,
        aliases=aliases,
        default_provider=default_provider,
        default_model=default_model,
        generation=gen,
        ui=ui,
        history=history,
        skills=skills,
        tools=tools,
    )
