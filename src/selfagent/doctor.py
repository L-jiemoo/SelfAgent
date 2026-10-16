"""Preflight checks, for when something is not working.

This program is expected to be plugged into machines nobody has seen before, so
"it does not work" is the normal case rather than the exception. Each check
answers one question a person would actually ask, in the order they would ask
them, and reports a single line.

The reachability probe stops at the HTTP layer on purpose. A real completion
would be the strongest possible check, but it costs money and several seconds
every time someone runs it, and the failures people actually hit -- wrong
base_url, no route, a proxy that eats the connection -- all show up here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .secrets import SecretStore
from .skills import Skill

TIMEOUT_SECONDS = 5.0


@dataclass
class Check:
    label: str
    ok: bool
    detail: str


def _short(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return " ".join(text.split())[:140]


def _reachable(name: str, base_url: str) -> Check:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a hard dependency
        return Check(name, False, f"缺少 httpx:{exc}")

    try:
        response = httpx.get(base_url, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    except Exception as exc:
        return Check(name, False, f"{base_url} 连不上:{_short(exc)}")

    # Any HTTP answer at all proves DNS, TLS and the route work. The status code
    # itself is uninteresting: a bare base_url routinely 404s by design.
    return Check(name, True, f"{base_url} 可达(HTTP {response.status_code})")


def run(
    config: Config,
    secrets: SecretStore,
    *,
    workspace: Path | None,
    skills: list[Skill],
    skills_root: Path,
) -> list[Check]:
    checks = [Check("config.json", True, str(config.path))]

    configured = secrets.providers
    checks.append(
        Check(
            "API Key",
            bool(configured),
            ", ".join(configured) if configured else "一个都没配,先运行 agent --setup",
        )
    )

    if workspace is None:
        checks.append(Check("工作目录", False, "本次启动未启用文件工具"))
    elif not workspace.is_dir():
        checks.append(Check("工作目录", False, f"不是目录:{workspace}"))
    else:
        writable = os.access(workspace, os.W_OK)
        checks.append(
            Check(
                "工作目录",
                writable,
                f"{workspace}({'可读可写' if writable else '只读,无法写文件'})",
            )
        )

    if skills_root.is_dir():
        checks.append(Check("技能目录", True, f"{skills_root}({len(skills)} 个技能)"))
    else:
        checks.append(Check("技能目录", True, f"{skills_root}(不存在,相当于没有技能)"))

    if not configured:
        checks.append(Check("网络", False, "没有 Key 可测"))
    for name in configured:
        spec = config.providers.get(name)
        if spec is None:
            checks.append(Check(name, False, "config.json 里已经没有这个服务商了"))
            continue
        checks.append(_reachable(name, spec.base_url))

    return checks
