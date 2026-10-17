"""Startup splash: logo, what this thing is, and what it can currently reach.

The infinity mark is a rasterised lemniscate of Bernoulli rather than
hand-drawn ASCII, so the two lobes are actually symmetric and the strands cross
at a single point. Regenerate it with a half-block rasteriser if you ever want a
different size; do not nudge the characters by hand.

Glyphs here are deliberately restricted to characters that survive a GBK
codepage, because a Chinese Windows console without chcp 65001 cannot encode
much of the block-drawing range.
"""

from __future__ import annotations

from typing import Mapping

from .render import GRADIENT_BOTTOM, GRADIENT_TOP

# Cyan -> violet by default; config.json's ui.theme overrides both ends.
COLOR_TOP = GRADIENT_TOP
COLOR_BOTTOM = GRADIENT_BOTTOM

LOGO: tuple[str, ...] = (
    "  ▄▄███▀▀▀▀███▄▄            ▄▄███▀▀▀▀███▄▄  ",
    "▄██▀           ▀▀█▄▄    ▄▄█▀▀           ▀██▄",
    "█▀                ▀▀█▄▄█▀▀                ▀█",
    "█▄                ▄▄█▀▀█▄▄                ▄█",
    "▀██▄           ▄▄█▀▀    ▀▀█▄▄           ▄██▀",
    "  ▀▀███▄▄▄▄███▀▀            ▀▀███▄▄▄▄███▀▀  ",
)

TITLE = "{name} v{version} · 便携式多模型 CLI Agent"
TAGLINE = "U盘即插即用,目标电脑无需安装 Python 或配置环境变量"
MODELS_HEADER = "已挂载模型      * = 已配置 Key,可直接使用"
NO_MODELS = "(未列模型,可任意指定)"
HINTS = "/help 命令    /model 换模型    /key 录入 Key    /exit 退出"
SKILLS_PREFIX = "已装技能  "
# Keeps the line inside a standard 80-column terminal once the prefix is added.
SKILLS_MAX = 68

# (text, rich style) -- style "" means terminal default
InfoLine = tuple[str, str]


def info_lines(
    version: str,
    current: str,
    directory: str,
    providers: Mapping[str, object],
    mounted: set[str],
    *,
    directory_label: str = "运行目录",
    tools_note: str | None = None,
    skills: list[str] | None = None,
    name: str = "SelfAgent",
) -> list[InfoLine]:
    """`providers` maps name -> anything with a `.models` list. `mounted` is the
    set of provider names that actually have an API key, which is the honest
    meaning of "mounted": configured but keyless cannot be called.

    `name` is the user's display name for the agent (config.json agent.name).
    """
    lines: list[InfoLine] = [
        (TITLE.format(name=name, version=version), "bold"),
        (TAGLINE, "dim"),
        ("", ""),
        (MODELS_HEADER, "bold"),
    ]

    for name, spec in providers.items():
        models = getattr(spec, "models", None) or []
        listing = ", ".join(models) if models else NO_MODELS
        if name in mounted:
            lines.append((f"  * {name:<9} {listing}", ""))
        else:
            lines.append((f"    {name:<9} {listing}", "dim"))

    lines += [
        ("", ""),
        (f"当前模型  {current}", "dim"),
        (f"{directory_label}  {directory}", "dim"),
    ]
    if tools_note:
        # Worth stating plainly at startup: this is the one thing the user needs
        # to know before letting the model near their files.
        lines.append((f"文件工具  {tools_note}", "dim"))
    if skills:
        listing = ", ".join(skills)
        if len(listing) > SKILLS_MAX:
            listing = listing[: SKILLS_MAX - 1].rstrip() + "…"
        lines.append((f"{SKILLS_PREFIX}{listing}", "dim"))
    lines += [
        ("", ""),
        (HINTS, "dim"),
    ]
    return lines
