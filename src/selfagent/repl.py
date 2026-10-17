"""Interactive mode.

Deliberately built on input() and one process-wide loop rather than a TUI
framework. A full-screen app is another frozen dependency and another thing that
can fail on a stranger's locked-down machine; for a chat loop it buys nothing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import __version__, banner, doctor, history, paths, registry
from .config import parse_color
from .engine import Engine
from .errors import SelfAgentError
from .render import Renderer, read_secret, summarize_usage

HELP = """命令(按用途分组):

对话
  /clear             清空当前上下文(保留本次会话)
  /compact           把当前对话压缩成摘要以省 token(会调用一次模型)
  /resume [序号]     列出历史会话,或载入某个继续对话(历史存在程序目录下)
  /forget [序号|区间|all] 删除历史会话记录(先列清单,再问一次才删)
  /export [路径]     把当前对话导出成 Markdown
  /model [名称]      查看或切换模型,例:/model claude、/model deepseek/deepseek-v4-pro
  /retry             重发上一条问题(网络抖了的时候用)
  /usage  /cost      查看本次会话的 token 用量与对话轮数

环境
  /status            一屏看清当前状态
  /doctor            自检:配置、Key、网络、工作目录、技能
  /config            显示 config.json 路径
  /skills            查看已安装的技能

文件工具
  /tools [on|off|readonly] 开关文件读写工具,或切成只读
  /dir [路径]        查看或切换工具的工作目录
  /undo              撤回上一次文件修改(会先给你看差异再问)
  /audit             本次会话改过哪些文件

自定义(写入 config.json,下次插到别的电脑也还在)
  /name <名字>       改界面显示名,立即生效
  /theme <上> <下> [强调色]  改渐变配色,例:/theme #22d3ee #a855f7
  /persona <文字>    一句话角色定位,下次启动生效(用 - 清空)
  /instructions <文字> 附加要求,下次启动生效(用 - 清空)
  /default-model <服务商/模型>  设为默认模型并保存

密钥
  /key <服务商>      录入或更新 API Key
  /keys              查看哪些服务商已配置 Key

其它
  /help              显示本帮助
  /exit  /quit       退出

不带斜杠的内容会直接发给模型。Ctrl+C 中断当前回答,Ctrl+D 退出。

文件工具怎么工作:
  模型可以读取和修改工作目录里的文件(启动画面上有显示)。每次写入都会先把
  差异列出来、等你敲 y 才落盘;工作目录之外的路径一律拒绝。让它做点需要动手的
  事情时说清楚文件名效果更好,例如「把 config.py 里的 timeout 改成 60 并说明改动」。

  单次模式默认只读,要允许写入请加 --allow-write;用 --no-tools 可完全关闭,
  用 --dir 路径 可指定别的工作目录。

  写错了可以用 /undo 撤回(撤销本身也要你确认)。/audit 能一屏看清这次会话动过
  哪些文件——在别人的电脑上用完,这一条尤其有用。凭据类文件(.env、*.pem、
  .git/ 等)默认拒写,可以在 config.json 里用 tools.protect 关掉。

技能:
  程序目录下的 skills\\ 里,每个技能是一个文件夹、内含 SKILL.md。模型只会读到
  技能的名字和描述,觉得切题时才用 read_file 打开全文。技能是只读的,而且不能
  执行命令、不能联网。装技能 = 把技能文件夹拷进 skills\\。"""


def _status_text(engine: Engine) -> str:
    toolbox = engine.toolbox
    if toolbox is None:
        tools = "未启用(去掉 --no-tools 重启可开启)"
    elif not engine.tools_enabled:
        tools = "本次会话已关闭"
    else:
        mode = "可读可写,写入前确认" if toolbox.allow_write else "只读"
        tools = f"{toolbox.root}  ({mode})"

    usage = engine.session.usage
    history = engine.session.history_path
    return "\n".join(
        [
            f"当前模型    {engine.target.label}",
            f"文件工具    {tools}",
            f"已装技能    {len(engine.skills)} 个",
            f"项目上下文  {engine.context_path or '(未找到 ' + engine.config.context.file + ')'}",
            f"本次改动    {len(engine.audit())} 个文件(/audit 查看)",
            f"对话轮数    {engine.session.turns}",
            f"token      输入 {usage.input_tokens} / 输出 {usage.output_tokens}",
            f"历史文件    {history if history else '(未启用)'}",
            f"配置文件    {engine.config.path}",
            f"已配 Key    {', '.join(engine.secrets.providers) or '无'}",
        ]
    )


def _format_started(raw: str) -> str:
    # The log stores an ISO timestamp; slicing beats parsing, because a mangled
    # value should still print rather than raise inside a listing command.
    return raw.replace("T", " ")[:16] or "(未知时间)"


def _session_listing(
    sessions: list[history.SessionSummary],
    directory: Path,
    header: str,
    *,
    with_name: bool = False,
) -> str:
    """`with_name` is for the delete listing: when the action cannot be undone,
    the file about to disappear is worth naming even at the cost of a wider
    line, because it can be checked against the folder by eye."""
    lines = [f"{header}(目录:{directory}):"]
    for index, item in enumerate(sessions, 1):
        name = f"{item.path.name}  " if with_name else ""
        lines.append(
            f"  {index:>2}  {_format_started(item.started)}  {name}"
            f"{item.target or '?':<28} {item.turns:>3} 轮  「{item.preview}」"
        )
    return "\n".join(lines)


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _pick_session(directory: Path, argument: str, renderer: Renderer) -> Path | None:
    if argument.isdigit():
        sessions = history.list_sessions(directory)
        index = int(argument)
        if not 1 <= index <= len(sessions):
            renderer.warning(
                f"序号超出范围:当前只列出 {len(sessions)} 个会话。先运行 /resume 看列表。"
            )
            return None
        return sessions[index - 1].path

    candidate = Path(argument)
    if not candidate.is_absolute():
        name = argument if argument.endswith(".jsonl") else f"{argument}.jsonl"
        candidate = directory / name
    if not candidate.is_file():
        renderer.warning(f"找不到会话文件:{candidate}")
        return None
    return candidate


def _resume(engine: Engine, renderer: Renderer, argument: str) -> None:
    if not engine.config.history.enabled:
        renderer.warning("config.json 里 history.enabled 为 false,没有会话记录可恢复。")
        return

    directory = paths.resolve_data_dir(engine.config.history.dir)

    if not argument:
        sessions = history.list_sessions(directory)
        if not sessions:
            renderer.status(f"没有找到历史会话(目录:{directory})。")
            return
        renderer.status(
            _session_listing(sessions, directory, "最近的会话")
            + "\n\n用法:/resume <序号> 载入并继续;也可以给文件名。"
        )
        return

    path = _pick_session(directory, argument, renderer)
    if path is None:
        return

    messages = history.load(path)
    if not messages:
        renderer.warning(f"{path.name} 里没有可用的对话内容。")
        return

    replaced = len(engine.session.messages)
    engine.session.restore(messages)
    note = ""
    if replaced and engine.session.history_path is not None:
        note = f"\n原对话的 {replaced} 条消息仍保存在 {engine.session.history_path}"
    renderer.status(
        f"已载入 {path.name}:{len(messages)} 条消息,{engine.session.turns} 轮。{note}"
    )


def _expand_indexes(tokens: list[str], count: int, renderer: Renderer) -> list[int] | None:
    """Turn index tokens into a sorted, de-duplicated list of indexes.

    Accepts `3`, `2-6` and any mix of the two. Sorted rather than in the order
    typed, because the result is shown back as a confirmation list and scanning
    it against the numbering just printed is the whole point.

    Everything is validated before anything is returned: a command that deletes
    files must not be half understood.
    """
    chosen: set[int] = set()
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= count:
                renderer.warning(
                    f"序号 {index} 超出范围:只有 {count} 个会话。先运行 /forget 看列表。"
                )
                return None
            chosen.add(index)
            continue

        start, separator, end = token.partition("-")
        if not separator or not start.isdigit() or not end.isdigit():
            renderer.warning(
                f"看不懂的序号 '{token}'。用法:/forget <序号|区间> ...,"
                "例如 /forget 1 3 5、/forget 2-6;也可以 /forget all"
            )
            return None
        low, high = int(start), int(end)
        if low > high:
            # `5-2` has exactly one sensible reading, so accept it rather than
            # make the user retype.
            low, high = high, low
        if low < 1 or high > count:
            renderer.warning(
                f"区间 {token} 超出范围:只有 {count} 个会话(合法序号 1-{count})。"
                "先运行 /forget 看列表。"
            )
            return None
        chosen.update(range(low, high + 1))

    return sorted(chosen)


def _select_sessions(
    sessions: list[history.SessionSummary],
    argument: str,
    renderer: Renderer,
    current: Path | None,
) -> list[history.SessionSummary]:
    """Turn "/forget 1 3 5" or "/forget 2-6" or "/forget all" into the sessions
    to delete.

    Never returns the log this session is still writing to. Deleting it would
    appear to work and then silently reappear on the next append, which is a
    worse outcome than refusing.
    """
    tokens = argument.split()
    if len(tokens) == 1 and tokens[0].lower() in ("all", "*"):
        indexes = list(range(1, len(sessions) + 1))
    else:
        indexes = _expand_indexes(tokens, len(sessions), renderer)
        if indexes is None:
            return []

    chosen = [sessions[index - 1] for index in indexes]
    current_name = current.name if current is not None else None
    kept = [item for item in chosen if item.path.name != current_name]
    if len(kept) != len(chosen):
        renderer.warning("当前会话自己的记录已跳过(它还在写,删掉会立刻重建)。")
    return kept


def _forget(engine: Engine, renderer: Renderer, argument: str) -> None:
    """Delete history logs. Destructive, so it always lists first and asks."""
    if not engine.config.history.enabled:
        renderer.warning("config.json 里 history.enabled 为 false,没有会话记录可清理。")
        return

    directory = paths.resolve_data_dir(engine.config.history.dir)
    sessions = history.list_sessions(directory, limit=None)

    if not sessions:
        renderer.status(f"没有可清理的会话(目录:{directory})。")
        return

    if not argument:
        renderer.status(
            _session_listing(sessions, directory, "可清理的会话")
            + "\n\n用法:/forget <序号|区间> ...,例如 1 3 5 或 2-6;/forget all 删除全部。"
            + "\n删除不可撤销,所以每次都会先把清单列出来再问你。"
        )
        return

    targets = _select_sessions(sessions, argument, renderer, engine.session.history_path)
    if not targets:
        return

    total = sum(history.size_of(item.path) for item in targets)
    renderer.warning(
        _session_listing(
            targets,
            directory,
            f"将要删除 {len(targets)} 个会话,共 {_human_size(total)}",
            with_name=True,
        )
    )
    if not renderer.ask_permission("确认删除?删掉就找不回来了"):
        renderer.status("已取消,什么都没删。")
        return

    reclaimed = 0
    deleted = 0
    for item in targets:
        size = history.size_of(item.path)
        if history.remove(item.path):
            reclaimed += size
            deleted += 1

    if deleted != len(targets):
        renderer.warning(f"{len(targets) - deleted} 个没能删掉(可能已不存在,或U盘只读)。")
    renderer.status(f"已删除 {deleted} 个会话,释放 {_human_size(reclaimed)}。")


def _to_markdown(engine: Engine) -> str:
    lines = [
        f"# {engine.config.agent.name} 对话导出",
        "",
        f"- 模型:{engine.target.label}",
        f"- 导出时间:{datetime.now().isoformat(timespec='seconds')}",
        f"- 对话轮数:{engine.session.turns}",
        "",
    ]
    for message in engine.session.messages:
        if message.tool_results:
            for result in message.tool_results:
                lines.append(f"**工具结果**{' —— 失败' if result.is_error else ''}")
                lines.append("")
                lines.append("```")
                lines.append(result.content)
                lines.append("```")
                lines.append("")
        elif message.tool_calls:
            names = ", ".join(f"`{call.name}`" for call in message.tool_calls)
            lines.append(f"**模型请求调用工具**:{names}")
            lines.append("")
            if message.content:
                lines.append(message.content)
                lines.append("")
        else:
            lines.append("## 你" if message.role == "user" else "## 模型")
            lines.append("")
            lines.append(message.content)
            lines.append("")
    return "\n".join(lines)


def _export(engine: Engine, renderer: Renderer, argument: str) -> None:
    """Write the current conversation out as Markdown.

    Straight to the filesystem rather than through the tools: the user asked for
    this by name, so it is authorised by the command itself, and the workspace
    sandbox has nothing to say about where a transcript should land.
    """
    if not engine.session.messages:
        renderer.warning("当前没有对话内容可导出。")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if argument:
        target = Path(argument).expanduser()
        if target.is_dir():
            target = target / f"conversation-{stamp}.md"
    else:
        target = Path.cwd() / f"conversation-{stamp}.md"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_to_markdown(engine), encoding="utf-8")
    except OSError as exc:
        renderer.error(f"导出失败:{exc.strerror or exc}")
        return
    renderer.status(f"已导出 {len(engine.session.messages)} 条消息到 {target}")


def _prompt_for_secret(label: str) -> str:
    return read_secret(label)


def _raw_section(config, name: str) -> dict:
    """The mutable raw section for a config key, created when missing."""
    section = config.raw.get(name)
    if not isinstance(section, dict):
        section = {}
        config.raw[name] = section
    return section


def _color_text(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _theme_command(engine: Engine, renderer: Renderer, argument: str) -> None:
    config = engine.config
    if not argument:
        renderer.status(
            "当前配色: top %s  bottom %s  accent %s\n"
            "用法:/theme <上色> <下色> [强调色],例如 /theme #22d3ee #a855f7"
            % (
                _color_text(config.ui.theme_top),
                _color_text(config.ui.theme_bottom),
                _color_text(config.ui.theme_accent),
            )
        )
        return

    parts = argument.split()
    if len(parts) not in (2, 3):
        renderer.warning("用法:/theme <上色> <下色> [强调色],颜色可写 #RRGGBB 或 #RGB。")
        return

    top = parse_color(parts[0], "theme.top")
    bottom = parse_color(parts[1], "theme.bottom")
    accent = parse_color(parts[2], "theme.accent") if len(parts) == 3 else top

    theme = _raw_section(config, "ui").get("theme")
    if not isinstance(theme, dict):
        theme = {}
        _raw_section(config, "ui")["theme"] = theme
    theme["top"] = _color_text(top)
    theme["bottom"] = _color_text(bottom)
    theme["accent"] = _color_text(accent)
    config.save()

    config.ui.theme_top = top
    config.ui.theme_bottom = bottom
    config.ui.theme_accent = accent
    renderer.set_theme(top, bottom, accent)
    renderer.status("配色已保存并立即生效。")


def _handle_command(engine: Engine, renderer: Renderer, line: str) -> bool:
    """Returns False when the user asked to quit."""
    command, _, argument = line[1:].partition(" ")
    command = command.strip().lower()
    argument = argument.strip()

    if command in ("exit", "quit", "q"):
        return False

    if command == "help":
        renderer.status(HELP)

    elif command == "model":
        if not argument:
            renderer.status(f"当前模型: {engine.target.label}\n\n{registry.describe(engine.config)}")
        else:
            target = engine.switch(argument)
            renderer.status(f"已切换到 {target.label}")

    elif command in ("default-model", "default_model"):
        if not argument:
            renderer.status(
                f"默认模型: {engine.config.default_provider}/{engine.config.default_model}\n"
                "用法:/default-model <服务商/模型>,例如 /default-model deepseek/deepseek-v4-pro"
            )
        else:
            target = engine.switch(argument)
            engine.config.default_provider = target.provider.name
            engine.config.default_model = target.model
            engine.config.raw["default_provider"] = target.provider.name
            engine.config.raw["default_model"] = target.model
            engine.config.save()
            renderer.status(f"默认模型已保存为 {target.label}(本次会话也已切换)。")

    elif command == "theme":
        _theme_command(engine, renderer, argument)

    elif command == "name":
        if not argument:
            renderer.status(f"当前名字: {engine.config.agent.name}")
        else:
            _raw_section(engine.config, "agent")["name"] = argument
            engine.config.save()
            engine.config.agent.name = argument
            renderer.set_agent_name(argument)
            renderer.status(
                f"界面名称已改为 {argument};系统提示里的身份下次启动生效。"
            )

    elif command in ("persona", "instructions"):
        current = getattr(engine.config.agent, command)
        if not argument:
            renderer.status(f"当前 {command}: {current or '(未设置)'}")
        else:
            value = "" if argument == "-" else argument
            _raw_section(engine.config, "agent")[command] = value
            engine.config.save()
            setattr(engine.config.agent, command, value)
            if value:
                renderer.status(f"{command} 已保存,下次启动生效。")
            else:
                renderer.status(f"{command} 已清空,下次启动生效。")

    elif command == "clear":
        engine.session.clear()
        renderer.status("上下文已清空。")

    elif command == "compact":
        renderer.status("正在压缩上下文…")
        summary = engine.compact()
        renderer.status(f"已压缩:用 {len(summary)} 字的摘要替换了原对话。")

    elif command == "resume":
        _resume(engine, renderer, argument)

    elif command == "forget":
        _forget(engine, renderer, argument)

    elif command == "undo":
        if engine.toolbox is None:
            renderer.warning("本次启动没有文件工具,没有可撤销的修改。")
        else:
            renderer.status(engine.undo() or "本次会话没有可撤销的修改了。")

    elif command == "audit":
        records = engine.audit()
        if not records:
            renderer.status("本次会话还没有修改过任何文件。")
        else:
            lines = [f"本次会话修改过 {len(records)} 个文件(最新在前):"]
            for index, record in enumerate(records, 1):
                kind = record.action if record.existed else "新建"
                lines.append(f"  {index:>2}  {kind:<12} {engine.toolbox.label(record.path)}")
            lines.append("")
            lines.append("用 /undo 可以逐个撤回(会先给你看差异再问)。")
            renderer.status("\n".join(lines))

    elif command == "export":
        _export(engine, renderer, argument)

    elif command == "key":
        provider_name, _, value = argument.partition(" ")
        provider_name = provider_name.strip()
        if not provider_name:
            renderer.warning("用法:/key <服务商> [key],不填 key 会提示输入。")
        elif provider_name not in engine.config.providers:
            renderer.warning(
                f"未知服务商 '{provider_name}'。可用:"
                f"{', '.join(sorted(engine.config.providers))}"
            )
        else:
            value = value.strip() or _prompt_for_secret(
                f"{provider_name} 的 API Key(输入不显示): "
            )
            if not value:
                renderer.warning("未输入 Key,已取消。")
            else:
                engine.secrets.set(provider_name, value)
                engine.secrets.save()
                renderer.status(f"已保存 {provider_name} 的 Key。")

    elif command == "keys":
        configured = engine.secrets.providers
        if configured:
            renderer.status(f"已配置 Key 的服务商:{', '.join(configured)}")
        else:
            renderer.status("尚未配置任何 Key,用 /key <服务商> 录入。")

    elif command == "skills":
        if not engine.skills:
            renderer.status(
                "尚未安装技能。把技能文件夹拷到程序目录下的 skills\\ 里即可"
                "(每个技能一个 SKILL.md)。"
            )
        else:
            lines = [f"已装 {len(engine.skills)} 个技能:"]
            for skill in engine.skills:
                lines.append(f"  {skill.name:<16} {skill.description}")
            renderer.status("\n".join(lines))

    elif command == "retry":
        renderer.status("重发上一条问题…")
        engine.retry()

    elif command in ("usage", "cost"):
        usage = summarize_usage(engine.session.usage) or "本次会话暂无用量数据"
        renderer.status(f"{usage}\n对话轮数 {engine.session.turns}")

    elif command == "status":
        renderer.status(_status_text(engine))

    elif command == "doctor":
        checks = doctor.run(
            engine.config,
            engine.secrets,
            workspace=engine.toolbox.root if engine.toolbox else None,
            skills=engine.skills,
            skills_root=paths.resolve_data_dir(engine.config.skills.dir),
        )
        lines = [f"{'OK' if check.ok else '!!'}  {check.label:<12} {check.detail}" for check in checks]
        lines.append("")
        lines.append("只测到能连上接口为止,不消耗 token。")
        renderer.status("\n".join(lines))

    elif command == "config":
        renderer.status(f"{engine.config.path}\n(可直接用记事本修改,改完重启程序生效)")

    elif command == "tools":
        wanted = argument.lower()
        if wanted in ("on", "off", "readonly"):
            renderer.status(engine.set_tools_mode(wanted))
        elif wanted:
            renderer.warning("用法:/tools on、/tools off 或 /tools readonly")
        elif engine.toolbox is None:
            renderer.status("文件工具:未启用(本次启动用了 --no-tools,或工作目录不存在)。")
        elif not engine.tools_enabled:
            renderer.status("文件工具:已关闭。用法:/tools on、/tools readonly、/tools off")
        else:
            state = "可读可写" if engine.toolbox.allow_write else "只读"
            renderer.status(f"文件工具:{state}。用法:/tools on、/tools readonly、/tools off")

    elif command == "dir":
        if engine.toolbox is None:
            renderer.warning("本次启动未启用文件工具,无法设置工作目录。")
        elif not argument:
            renderer.status(f"当前工作目录:{engine.toolbox.root}")
        else:
            target = Path(argument).expanduser()
            if not target.is_dir():
                renderer.warning(f"不是目录或不存在:{target}")
            else:
                engine.toolbox.root = target.resolve()
                renderer.status(
                    f"工作目录已切换到 {engine.toolbox.root}\n"
                    "注意:该目录里的文件现在可以被模型读写(写入仍需你确认)。"
                )

    else:
        renderer.warning(f"未知命令 '/{command}',输入 /help 查看全部命令。")

    return True


def run(engine: Engine, renderer: Renderer) -> int:
    toolbox = engine.toolbox
    if toolbox is not None:
        directory, label = str(toolbox.root), "工作目录"
        note = "可读可写,每次写入前确认" if toolbox.allow_write else "只读"
    else:
        directory, label, note = str(paths.app_dir()), "运行目录", None

    renderer.splash(
        banner.LOGO,
        banner.info_lines(
            __version__,
            engine.target.label,
            directory,
            engine.config.providers,
            set(engine.secrets.providers),
            directory_label=label,
            tools_note=note,
            skills=[skill.name for skill in engine.skills] or None,
            name=engine.config.agent.name,
        ),
    )

    while True:
        try:
            line = renderer.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line.startswith("/"):
            try:
                if not _handle_command(engine, renderer, line):
                    return 0
            except SelfAgentError as exc:
                renderer.error(str(exc))
            continue

        try:
            engine.ask(line)
        except KeyboardInterrupt:
            renderer.finish()
            renderer.warning("已中断。")
        except SelfAgentError as exc:
            renderer.finish()
            renderer.error(str(exc))
