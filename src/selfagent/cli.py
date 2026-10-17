"""Argument parsing and dispatch.

Mode selection is intentionally implicit: if a question was given on the command
line, or if stdin is a pipe, run once and exit. Otherwise start the REPL. That
means `agent "q"`, `echo q | agent`, and a bare `agent` all do what a shell user
expects without a mode flag.
"""

from __future__ import annotations

import argparse
import json
import locale
import sys
from pathlib import Path

from . import __version__, context, paths, registry, secrets, skills as skills_mod
from .config import Config, load_config
from .engine import Engine
from .errors import SelfAgentError
from .render import Renderer, read_secret
from .secrets import SecretStore
from .skills import Skill
from .tools import Toolbox, auto_root


def _display_name() -> str:
    """Best-effort display name for --version.

    Read straight from config.json rather than through load_config(): --version
    must work on a machine with a broken config, and it must not materialise a
    default config.json as a side effect of printing a version string.
    """
    try:
        raw = json.loads(paths.config_path().read_text(encoding="utf-8"))
        agent = raw.get("agent") if isinstance(raw, dict) else None
        name = agent.get("name") if isinstance(agent, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    except (OSError, ValueError):
        pass
    return "SelfAgent"


def build_parser(name: str = "SelfAgent") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="便携式多模型命令行助手。无参数进入交互模式,带问题则单次回答。",
        epilog="示例: agent \"你好\"  |  agent -m claude \"你好\"  |  type q.txt | agent",
    )
    parser.add_argument("prompt", nargs="*", help="要提问的内容;省略则进入交互模式")
    parser.add_argument("-m", "--model", help="模型、别名或 服务商/模型,如 claude、deepseek/deepseek-v4-pro")
    parser.add_argument("--setup", action="store_true", help="录入或更新 API Key(交互式)")
    parser.add_argument("--list", action="store_true", help="列出已配置的服务商与模型")
    parser.add_argument("--no-history", action="store_true", help="本次不写历史记录")
    parser.add_argument("--no-color", action="store_true", help="禁用彩色输出")
    parser.add_argument("--show-thinking", action="store_true", help="显示模型的思考过程")
    parser.add_argument(
        "--dir",
        metavar="路径",
        help="工具的工作目录(默认=当前目录)。所有读写都被限制在这个目录内。",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="允许单次模式写文件。不加则单次模式只能读;加上即代表授权,不再逐个确认。",
    )
    parser.add_argument("--no-tools", action="store_true", help="完全关闭文件读写工具")
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="本次不加载 skills 目录里的技能",
    )
    parser.add_argument("--version", action="version", version=f"{name} {__version__}")
    return parser


def _read_passphrase(label: str) -> str:
    return read_secret(label)


def _unlock(config: Config, renderer: Renderer, *, allow_create: bool) -> SecretStore | None:
    """Unlock the key store, prompting for the passphrase.

    allow_create is only true in interactive mode. Reading a passphrase requires
    a console, and in piped mode the console is not where the question came from
    -- so a missing store there is an error telling the user to run --setup,
    not a hang waiting on input that will never arrive.
    """
    if not secrets.store_exists():
        if not allow_create:
            renderer.error(
                "尚未配置 API Key。请先在交互模式下运行 `agent --setup` 录入。"
            )
            return None
        renderer.status("首次使用:请先设置一个口令来加密保存 API Key(仅需记住,不写入任何地方)。")
        passphrase = _read_passphrase("设置口令(输入不显示,回车确认): ")
        if not passphrase:
            renderer.error("口令不能为空。")
            return None
        confirm = _read_passphrase("再输一次: ")
        if passphrase != confirm:
            renderer.error("两次输入不一致。")
            return None
        store = secrets.create(passphrase)
        store.save()
        renderer.status(f"已创建密钥库。接着运行 `agent --setup` 录入 API Key。")
        return store

    for attempt in range(3):
        passphrase = _read_passphrase("口令(输入不显示,回车确认): ")
        if not passphrase:
            renderer.error("已取消。")
            return None
        try:
            return secrets.unlock(passphrase)
        except SelfAgentError as exc:
            renderer.error(str(exc))
            if attempt < 2:
                renderer.status(f"还可以重试 {2 - attempt} 次。")
    return None


def run_setup(config: Config, renderer: Renderer) -> int:
    store = _unlock(config, renderer, allow_create=True)
    if store is None:
        return 1

    renderer.status("可用服务商:" + ", ".join(sorted(config.providers)))
    renderer.status("逐个录入,直接回车结束。已有 Key 的服务商:" + (", ".join(store.providers) or "无"))

    while True:
        name = input("服务商名(回车结束)> ").strip()
        if not name:
            break
        if name not in config.providers:
            renderer.warning(f"未知服务商 '{name}'。")
            continue

        value = _read_passphrase(f"{name} 的 API Key(回车跳过): ")
        if not value:
            renderer.status(f"跳过 {name}。")
            continue
        store.set(name, value)
        renderer.status(f"已记录 {name}。")

    try:
        store.save()
    except SelfAgentError as exc:
        renderer.error(str(exc))
        return 1
    renderer.status("已保存到 " + str(paths.secrets_path()))
    return 0


def run_list(config: Config, renderer: Renderer) -> int:
    print(registry.describe(config))
    print()
    if secrets.store_exists():
        print("  secrets.enc 存在(需口令解锁后才能列出已配置的 Key)")
    else:
        print("  尚未创建 secrets.enc,运行 `agent --setup` 录入 Key")

    root = paths.resolve_data_dir(config.skills.dir)
    installed = skills_mod.discover(root) if config.skills.enabled else []
    print()
    if installed:
        print(f"  已装技能 {len(installed)} 个({root}):")
        for skill in installed:
            print(f"    {skill.name:<16} {skill.description}")
    else:
        print(f"  尚未安装技能。把技能文件夹拷到 {root} 下即可(每个技能一个 SKILL.md)。")
    return 0


def _make_confirm(renderer: Renderer, *, interactive: bool, prompt: bool):
    """Build the callback the toolbox uses to approve a write.

    Interactive sessions ask. Non-interactive ones do not have a terminal to ask
    on -- stdin is the question -- so reaching this point at all means the user
    passed --allow-write, which is the authorisation. The diff is still printed
    as a record of what changed.
    """

    def confirm(action: str, path, old_text: str, new_text: str) -> bool:
        renderer.diff(action, path, old_text, new_text)
        if interactive and prompt:
            return renderer.ask_permission(f"允许{action}?")
        return True

    return confirm


def _discover_skills(config: Config, args) -> tuple[Path, list[Skill]]:
    """Find the skills installed next to the program. Returns (root, skills).

    Discovery happens before the toolbox is built because the toolbox needs the
    skills folder as a read-only root. Whether the catalogue actually reaches the
    model is decided afterwards -- there is no point advertising skills it has no
    way to open.
    """
    if not config.skills.enabled or args.no_skills:
        return Path(), []
    root = paths.resolve_data_dir(config.skills.dir)
    return root, skills_mod.discover(root)


def _inject_catalogue(config: Config, installed: list[Skill], root: Path) -> None:
    """Append the skill index to the system prompt.

    The index, not the bodies: a skill costs a couple of lines per request until
    the model decides it is relevant and reads it.
    """
    block = skills_mod.catalog(installed, root)
    if block:
        config.generation.system = (config.generation.system or "") + block


def _build_toolbox(
    config: Config,
    args,
    renderer: Renderer,
    *,
    interactive: bool,
    read_roots: tuple[Path, ...] = (),
):
    """Returns None when tools are off, unavailable, or the root is missing."""
    if not config.tools.enabled or args.no_tools:
        return None

    root = auto_root(args.dir or config.tools.workspace)
    if not root.exists():
        renderer.warning(f"工作目录不存在:{root},已关闭文件工具。")
        return None

    # Interactive sessions may write (each write is confirmed). One-shot
    # sessions may not, unless the user said so explicitly.
    allow_write = interactive or args.allow_write

    return Toolbox(
        root=root,
        confirm=_make_confirm(
            renderer,
            interactive=interactive,
            prompt=config.tools.confirm_writes,
        ),
        allow_write=allow_write,
        max_read_bytes=config.tools.max_read_bytes,
        read_roots=read_roots,
        protect=config.tools.protect,
    )


def _configure_output_encoding() -> None:
    """Write UTF-8 when redirected; leave the console alone.

    On a Chinese Windows box, a redirected stream defaults to GBK, so
    `agent "q" > out.txt` would produce a file that only opens correctly on
    another GBK machine. Forcing UTF-8 there makes the output portable.

    A terminal is deliberately excluded: Python encodes console output using the
    active code page, which is already correct whether the user launched through
    agent.bat (chcp 65001) or double-clicked the exe (cp936). Overriding it would
    be what actually caused mojibake.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def _read_stdin_if_piped() -> str | None:
    if sys.stdin is None or sys.stdin.isatty():
        return None

    try:
        data = sys.stdin.buffer.read()
    except (OSError, AttributeError):
        try:
            return sys.stdin.read().strip() or None
        except (OSError, UnicodeDecodeError):
            return None

    if not data:
        return None

    # Question files are UTF-8 more often than not, but a notepad-saved file on
    # a Chinese system is GBK. Try both rather than pick one and mis-decode.
    for encoding in ("utf-8", locale.getpreferredencoding(False)):
        try:
            return data.decode(encoding).strip() or None
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace").strip() or None


def main(argv: list[str] | None = None) -> int:
    _configure_output_encoding()

    args = build_parser(_display_name()).parse_args(argv)

    renderer = Renderer(
        color=not args.no_color,
        show_thinking=args.show_thinking,
    )

    try:
        config = load_config()
    except SelfAgentError as exc:
        renderer.error(str(exc))
        return 1

    renderer.set_theme(
        config.ui.theme_top, config.ui.theme_bottom, config.ui.theme_accent
    )
    renderer.set_agent_name(config.agent.name)
    # Identity goes in before skills and project conventions, so the prompt
    # reads: base instruction -> who you are -> capabilities -> this project.
    identity = config.agent.system_block()
    if identity:
        config.generation.system = (config.generation.system or "") + identity

    # The Renderer is built before the config is read -- it is what reports a
    # broken config -- so the animation switch is applied here rather than in
    # the constructor.
    renderer.animate = renderer.animate and config.ui.splash_animation
    renderer.show_spinner = renderer.show_spinner and config.ui.spinner
    renderer.show_box = renderer.show_box and config.ui.prompt_box

    if args.show_thinking:
        config.ui.show_thinking = True

    if args.list:
        return run_list(config, renderer)

    if args.setup:
        return run_setup(config, renderer)

    piped = _read_stdin_if_piped()
    prompt = " ".join(args.prompt).strip() or piped
    interactive = prompt is None

    store = _unlock(config, renderer, allow_create=interactive)
    if store is None:
        return 1

    skills_root, installed = _discover_skills(config, args)

    # The project's own conventions, read from wherever the tools work. Done
    # regardless of whether tools are on: knowing the conventions is useful even
    # when the model is only talking.
    workspace = auto_root(args.dir or config.tools.workspace)
    context_block, context_path = ("", None)
    if config.context.enabled:
        context_block, context_path = context.load(
            workspace, config.context.file, config.context.max_chars
        )
    if context_block:
        config.generation.system = (config.generation.system or "") + context_block

    toolbox = _build_toolbox(
        config,
        args,
        renderer,
        interactive=interactive,
        read_roots=(skills_root,) if installed else (),
    )

    # Only advertise skills the model has a way to open.
    if toolbox is not None and installed:
        _inject_catalogue(config, installed, skills_root)

    try:
        engine = Engine(
            config,
            store,
            renderer,
            model_token=args.model,
            persist=not args.no_history,
            toolbox=toolbox,
            skills=installed,
            context_path=context_path,
        )
    except SelfAgentError as exc:
        renderer.error(str(exc))
        return 1

    if interactive:
        from . import repl

        return repl.run(engine, renderer)

    from . import oneshot

    return oneshot.run(engine, renderer, prompt)
