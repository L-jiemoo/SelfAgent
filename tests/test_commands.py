"""The Claude-CLI-inspired commands: /compact, /doctor, /status, /tools, /dir.

The load-bearing property here is that compaction is safe. A compaction that
loses the conversation is far worse than one that simply does not happen, so
most of these tests are about what survives a failure.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from selfagent import doctor, render, repl
from selfagent.config import DEFAULT_CONFIG, load_config
from selfagent.engine import MAX_TRANSCRIPT, Engine
from selfagent.errors import ConfigError, ProviderError, SelfAgentError
from selfagent.providers.base import Done, TextDelta, ToolCall, Usage
from selfagent.render import Renderer
from selfagent.secrets import SecretStore
from selfagent.session import COMPACT_MARKER
from selfagent.tools import Toolbox


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def make_engine(app_dir: Path, toolbox=None, *, keys=("deepseek",), persist=False) -> Engine:
    config = load_config()
    store = SecretStore(keys={name: "sk-test" for name in keys})
    renderer = Renderer(color=False, out=io.StringIO(), err=io.StringIO())
    return Engine(config, store, renderer, persist=persist, toolbox=toolbox)


def make_toolbox(root: Path) -> Toolbox:
    return Toolbox(root=root, confirm=lambda *args: True)


def stream(*events):
    return lambda _key: iter(events)


# ---- /compact --------------------------------------------------------


def test_compact_replaces_the_conversation(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("你好")
    engine.session.add_assistant("你好,有什么可以帮你")
    engine._stream = stream(TextDelta("摘要正文"), Usage(3, 4), Done())

    summary = engine.compact()

    assert summary == "摘要正文"
    assert [message.role for message in engine.session.messages] == ["user", "assistant"]
    assert engine.session.messages[0].content == COMPACT_MARKER
    assert engine.session.messages[1].content == "摘要正文"


def test_compacted_history_still_starts_with_a_user_turn(app_dir: Path) -> None:
    # Anthropic rejects a history that opens with an assistant turn, so the
    # synthetic prompt is not decoration -- it is what makes the request legal.
    engine = make_engine(app_dir)
    engine.session.add_user("q")
    engine._stream = stream(TextDelta("s"), Done())
    engine.compact()
    assert engine.session.messages[0].role == "user"


def test_compact_keeps_the_original_when_the_request_fails(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("你好")
    engine.session.add_assistant("在的")
    before = list(engine.session.messages)

    def boom(_key):
        raise ProviderError("网络炸了")

    engine._stream = boom

    with pytest.raises(ProviderError):
        engine.compact()

    assert engine.session.messages == before


def test_compact_keeps_the_original_when_the_model_says_nothing(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("你好")
    before = list(engine.session.messages)
    engine._stream = stream(Done())

    with pytest.raises(SelfAgentError, match="没有返回摘要"):
        engine.compact()

    assert engine.session.messages == before


def test_compact_disables_tools_for_the_summary_request(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("你好")
    engine.tools_enabled = True
    seen: list[bool] = []

    def fake(_key):
        seen.append(engine.tools_enabled)
        yield TextDelta("sum")
        yield Done()

    engine._stream = fake
    engine.compact()

    assert seen == [False], "a summariser that can edit files is not a summariser"
    assert engine.tools_enabled is True, "restored afterwards"


def test_compact_without_a_conversation_is_an_error(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    with pytest.raises(SelfAgentError, match="没有可压缩"):
        engine.compact()


def test_compact_without_a_key_is_an_error(app_dir: Path) -> None:
    engine = make_engine(app_dir, keys=())
    engine.session.add_user("你好")
    with pytest.raises(ProviderError, match="没有为"):
        engine.compact()


def test_transcript_labels_the_roles(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("问题")
    engine.session.add_assistant("回答")
    text = engine._transcript()
    assert "[user] 问题" in text
    assert "[assistant] 回答" in text


def test_a_huge_transcript_is_truncated_from_the_front(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("开头" + "X" * (MAX_TRANSCRIPT + 5000))
    text = engine._transcript()
    assert text.startswith("(更早的部分已省略)")
    assert len(text) <= MAX_TRANSCRIPT + 40


# ---- /tools ----------------------------------------------------------


def test_set_tools_without_a_toolbox_explains_itself(app_dir: Path) -> None:
    engine = make_engine(app_dir)
    with pytest.raises(SelfAgentError, match="没有文件工具"):
        engine.set_tools_mode("on")


def test_tool_modes_cycle_through_off_readonly_and_on(app_dir: Path, tmp_path: Path) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))
    assert engine.tools_enabled is True
    assert engine.toolbox.allow_write is True

    engine.set_tools_mode("readonly")
    assert engine.tools_enabled is True, "the model can still look around"
    assert engine.toolbox.allow_write is False, "but it cannot write"

    engine.set_tools_mode("off")
    assert engine.tools_enabled is False

    engine.set_tools_mode("on")
    assert engine.tools_enabled is True
    assert engine.toolbox.allow_write is True


# ---- /doctor ---------------------------------------------------------


def test_doctor_reports_the_basics(app_dir: Path, tmp_path: Path) -> None:
    checks = doctor.run(
        load_config(),
        SecretStore(keys={}),
        workspace=tmp_path,
        skills=[],
        skills_root=tmp_path / "skills",
    )
    by_label = {check.label: check for check in checks}

    assert by_label["config.json"].ok is True
    assert by_label["API Key"].ok is False
    assert "setup" in by_label["API Key"].detail
    assert by_label["工作目录"].ok is True
    # A missing skills folder is normal, not a fault.
    assert by_label["技能目录"].ok is True
    assert "不存在" in by_label["技能目录"].detail


def test_doctor_flags_a_missing_workspace(app_dir: Path, tmp_path: Path) -> None:
    checks = doctor.run(
        load_config(), SecretStore(keys={}), workspace=None, skills=[], skills_root=tmp_path
    )
    assert {check.label: check for check in checks}["工作目录"].ok is False


def test_doctor_probes_only_providers_that_have_keys(
    app_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []

    def fake(name: str, base_url: str) -> doctor.Check:
        probed.append(name)
        return doctor.Check(name, True, base_url)

    monkeypatch.setattr(doctor, "_reachable", fake)
    doctor.run(
        load_config(),
        SecretStore(keys={"deepseek": "k"}),
        workspace=None,
        skills=[],
        skills_root=app_dir / "skills",
    )
    assert probed == ["deepseek"]


def test_doctor_notes_a_key_for_a_provider_that_vanished(app_dir: Path) -> None:
    checks = doctor.run(
        load_config(),
        SecretStore(keys={"ghost": "k"}),
        workspace=None,
        skills=[],
        skills_root=app_dir / "skills",
    )
    assert {check.label: check for check in checks}["ghost"].ok is False


def test_reachability_failure_is_reported_not_raised() -> None:
    # Nothing listens on port 1; the probe must come back as a failed Check.
    check = doctor._reachable("x", "http://127.0.0.1:1/")
    assert check.ok is False
    assert check.detail


# ---- command dispatch ------------------------------------------------


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> Renderer:
    monkeypatch.setattr(render, "_isatty", lambda stream: True)
    return Renderer(color=True, out=Tty(), err=io.StringIO())


def test_status_prints_the_essentials(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    assert repl._handle_command(engine, console, "/status") is True

    text = console.out.getvalue()
    assert "当前模型" in text
    assert "文件工具" in text
    assert "已装技能" in text
    assert "对话轮数" in text


def test_cost_is_an_alias_for_usage(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/cost")
    assert "对话轮数" in console.out.getvalue()


def test_tools_with_no_argument_reports_the_state(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/tools")
    assert "未启用" in console.out.getvalue()


def test_tools_off_then_on(app_dir: Path, tmp_path: Path, console: Renderer) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))

    repl._handle_command(engine, console, "/tools off")
    assert engine.tools_enabled is False
    assert "关闭" in console.out.getvalue()

    repl._handle_command(engine, console, "/tools on")
    assert engine.tools_enabled is True


def test_tools_rejects_a_bad_argument(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/tools maybe")
    assert "用法" in console.err.getvalue()


def test_dir_switches_the_workspace(app_dir: Path, tmp_path: Path, console: Renderer) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    engine = make_engine(app_dir, toolbox=make_toolbox(workspace))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    repl._handle_command(engine, console, f"/dir {elsewhere}")

    assert engine.toolbox.root == elsewhere.resolve()
    assert "已切换" in console.out.getvalue()


def test_dir_refuses_a_path_that_is_not_a_directory(
    app_dir: Path, tmp_path: Path, console: Renderer
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    engine = make_engine(app_dir, toolbox=make_toolbox(workspace))

    repl._handle_command(engine, console, f"/dir {tmp_path / 'nope'}")

    assert engine.toolbox.root == workspace.resolve()
    assert "不是目录" in console.err.getvalue()


def test_unknown_command_is_reported(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/nonsense")
    assert "未知命令" in console.err.getvalue()


def test_help_lists_every_command(app_dir: Path, console: Renderer) -> None:
    for name in (
        "/status", "/doctor", "/compact", "/tools", "/dir", "/cost",
        "/resume", "/forget", "/undo", "/audit", "/export", "/retry",
        "/theme", "/name", "/persona", "/instructions", "/default-model",
    ):
        assert name in repl.HELP


# ---- customisation commands write config.json ------------------------


def read_saved(app_dir: Path) -> dict:
    return json.loads((app_dir / "config.json").read_text(encoding="utf-8"))


def test_theme_command_saves_and_applies(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/theme #ff0000 #00ff00 #0000ff")

    saved = read_saved(app_dir)
    assert saved["ui"]["theme"] == {
        "top": "#ff0000",
        "bottom": "#00ff00",
        "accent": "#0000ff",
    }
    assert engine.config.ui.theme_top == (255, 0, 0)
    assert console.gradient_top == (255, 0, 0)
    assert console.gradient_bottom == (0, 255, 0)
    assert console.accent == (0, 0, 255)


def test_theme_accent_follows_the_top_when_omitted(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/theme f00 00f")

    assert read_saved(app_dir)["ui"]["theme"] == {
        "top": "#ff0000",
        "bottom": "#0000ff",
        "accent": "#ff0000",
    }
    assert console.accent == (255, 0, 0)


def test_theme_rejects_a_bad_colour_without_writing(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    before = (app_dir / "config.json").read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="theme.top"):
        repl._handle_command(engine, console, "/theme nope #00ff00")

    assert (app_dir / "config.json").read_text(encoding="utf-8") == before


def test_name_command_saves_and_renames(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/name 小助手")

    assert read_saved(app_dir)["agent"]["name"] == "小助手"
    assert engine.config.agent.name == "小助手"
    assert console.agent_name == "小助手"


def test_persona_and_instructions_are_saved_for_the_next_launch(
    app_dir: Path, console: Renderer
) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/persona 严谨的代码审阅助手")
    repl._handle_command(engine, console, "/instructions 回答先给结论")

    saved = read_saved(app_dir)
    assert saved["agent"]["persona"] == "严谨的代码审阅助手"
    assert saved["agent"]["instructions"] == "回答先给结论"
    assert "下次启动" in console.out.getvalue()


def test_persona_dash_clears_it(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/persona 严谨的代码审阅助手")
    repl._handle_command(engine, console, "/persona -")

    assert read_saved(app_dir)["agent"]["persona"] == ""
    assert engine.config.agent.persona == ""


def test_default_model_is_saved_and_switched(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)

    repl._handle_command(engine, console, "/default-model deepseek/deepseek-v4-pro")

    saved = read_saved(app_dir)
    assert saved["default_provider"] == "deepseek"
    assert saved["default_model"] == "deepseek-v4-pro"
    assert engine.target.model == "deepseek-v4-pro"


# ---- /undo and /audit ------------------------------------------------


def test_undo_without_a_toolbox_says_so(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/undo")
    assert "没有文件工具" in console.err.getvalue()


def test_undo_reports_when_there_is_nothing_to_take_back(
    app_dir: Path, tmp_path: Path, console: Renderer
) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))
    repl._handle_command(engine, console, "/undo")
    assert "没有可撤销" in console.out.getvalue()


def test_audit_reports_an_untouched_session(
    app_dir: Path, tmp_path: Path, console: Renderer
) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))
    repl._handle_command(engine, console, "/audit")
    assert "还没有修改过" in console.out.getvalue()


def test_audit_lists_what_was_written(app_dir: Path, tmp_path: Path, console: Renderer) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))
    engine.toolbox.execute(
        ToolCall("c1", "write_file", {"path": "note.txt", "content": "hi"})
    )

    repl._handle_command(engine, console, "/audit")

    text = console.out.getvalue()
    assert "修改过 1 个文件" in text
    assert "note.txt" in text


def test_undo_takes_the_write_back(app_dir: Path, tmp_path: Path, console: Renderer) -> None:
    engine = make_engine(app_dir, toolbox=make_toolbox(tmp_path))
    engine.toolbox.execute(
        ToolCall("c1", "write_file", {"path": "note.txt", "content": "hi"})
    )

    repl._handle_command(engine, console, "/undo")

    assert not (tmp_path / "note.txt").exists()
    assert "删除" in console.out.getvalue()


# ---- /export ---------------------------------------------------------


def test_export_writes_a_markdown_file(app_dir: Path, tmp_path: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    engine.session.add_user("问题")
    engine.session.add_assistant("回答")
    target = tmp_path / "out.md"

    repl._handle_command(engine, console, f"/export {target}")

    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "问题" in body and "回答" in body
    assert "已导出" in console.out.getvalue()


def test_export_title_uses_the_configured_agent_name(
    app_dir: Path, tmp_path: Path, console: Renderer
) -> None:
    data = json.loads(json.dumps(DEFAULT_CONFIG))
    data["agent"]["name"] = "小助手"
    (app_dir / "config.json").write_text(json.dumps(data), encoding="utf-8")

    engine = make_engine(app_dir)
    engine.session.add_user("问题")
    engine.session.add_assistant("回答")
    target = tmp_path / "out.md"

    repl._handle_command(engine, console, f"/export {target}")

    assert target.read_text(encoding="utf-8").startswith("# 小助手 对话导出")


def test_export_with_nothing_to_say(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/export")
    assert "没有对话内容" in console.err.getvalue()


# ---- /retry ----------------------------------------------------------


def test_retry_without_a_question_is_refused(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    with pytest.raises(SelfAgentError, match="没有可重发"):
        repl._handle_command(engine, console, "/retry")


# ---- /resume ---------------------------------------------------------


def write_log(app_dir: Path, name: str, *records) -> None:
    import json

    directory = app_dir / "history"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def a_session(when: str = "2026-10-16T15:30:00") -> list[dict]:
    return [
        {"type": "session_start", "time": when, "target": "deepseek/deepseek-v4-flash"},
        {"type": "user", "text": "帮我看看项目结构"},
        {"type": "assistant", "text": "好的"},
    ]


def test_resume_with_no_argument_lists_sessions(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/resume")

    text = console.out.getvalue()
    assert "最近的会话" in text
    assert "2026-10-16 15:30" in text
    assert "帮我看看项目结构" in text


def test_resume_reports_an_empty_history(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/resume")
    assert "没有找到历史会话" in console.out.getvalue()


def test_resume_loads_by_index(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/resume 1")

    assert [message.role for message in engine.session.messages] == ["user", "assistant"]
    assert engine.session.turns == 1
    assert "已载入" in console.out.getvalue()


def test_resume_rejects_an_index_that_is_out_of_range(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/resume 99")

    assert engine.session.messages == []
    assert "超出范围" in console.err.getvalue()


def test_resume_accepts_a_filename(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/resume 20261016-153000")

    assert engine.session.turns == 1


def test_resume_reports_a_missing_file(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/resume ghost")
    assert "找不到会话文件" in console.err.getvalue()


def test_resume_is_refused_when_history_is_off(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    engine.config.history.enabled = False
    repl._handle_command(engine, console, "/resume")
    assert "history.enabled" in console.err.getvalue()


# ---- /forget ---------------------------------------------------------


def approve(monkeypatch: pytest.MonkeyPatch, console: Renderer, answer: bool = True) -> None:
    monkeypatch.setattr(console, "ask_permission", lambda question: answer)


def test_forget_lists_sessions_and_deletes_nothing(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/forget")

    assert "可清理的会话" in console.out.getvalue()
    assert (app_dir / "history" / "20261016-153000.jsonl").exists()


def test_forget_deletes_after_confirmation(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget 1")

    assert not (app_dir / "history" / "20261016-153000.jsonl").exists()
    assert "已删除 1 个会话" in console.out.getvalue()


def test_forget_shows_what_it_is_about_to_delete(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget 1")

    assert "20261016-153000.jsonl" in console.err.getvalue()


def test_forget_deletes_nothing_when_refused(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())
    approve(monkeypatch, console, answer=False)

    repl._handle_command(engine, console, "/forget 1")

    assert (app_dir / "history" / "20261016-153000.jsonl").exists()
    assert "已取消" in console.out.getvalue()


def test_forget_all_clears_the_folder(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    for index in range(3):
        write_log(app_dir, f"2026100{index + 1}-000000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget all")

    assert list((app_dir / "history").glob("*.jsonl")) == []
    assert "已删除 3 个会话" in console.out.getvalue()


def test_forget_accepts_several_indexes(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261001-000000.jsonl", *a_session())
    write_log(app_dir, "20261002-000000.jsonl", *a_session())
    write_log(app_dir, "20261003-000000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget 1 3")

    remaining = sorted(item.name for item in (app_dir / "history").glob("*.jsonl"))
    assert remaining == ["20261002-000000.jsonl"]


def test_forget_rejects_an_out_of_range_index_without_deleting(
    app_dir: Path, console: Renderer
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/forget 99")

    assert "超出范围" in console.err.getvalue()
    assert (app_dir / "history" / "20261016-153000.jsonl").exists()


def test_forget_rejects_a_non_numeric_argument(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261016-153000.jsonl", *a_session())

    repl._handle_command(engine, console, "/forget yesterday")

    assert "看不懂" in console.err.getvalue()
    assert (app_dir / "history" / "20261016-153000.jsonl").exists()


def test_forget_never_touches_the_log_being_written(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir, persist=True)
    current = engine.session.history_path
    assert current is not None and current.exists()
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget all")

    assert current.exists(), "deleting it would just make it reappear on the next append"
    assert "已跳过" in console.err.getvalue()


def test_forget_reports_an_empty_folder(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    repl._handle_command(engine, console, "/forget")
    assert "没有可清理的会话" in console.out.getvalue()


def test_forget_is_refused_when_history_is_off(app_dir: Path, console: Renderer) -> None:
    engine = make_engine(app_dir)
    engine.config.history.enabled = False
    repl._handle_command(engine, console, "/forget")
    assert "history.enabled" in console.err.getvalue()


# ---- index ranges ----------------------------------------------------


@pytest.fixture
def quiet() -> Renderer:
    return Renderer(color=False, out=io.StringIO(), err=io.StringIO())


def test_a_range_expands_to_every_index_between(quiet: Renderer) -> None:
    assert repl._expand_indexes(["2-5"], 10, quiet) == [2, 3, 4, 5]


def test_a_lone_index_is_still_one_index(quiet: Renderer) -> None:
    assert repl._expand_indexes(["3"], 10, quiet) == [3]


def test_indexes_and_ranges_can_be_mixed(quiet: Renderer) -> None:
    assert repl._expand_indexes(["1", "3-5", "8"], 10, quiet) == [1, 3, 4, 5, 8]


def test_a_reversed_range_is_accepted(quiet: Renderer) -> None:
    # "5-2" has exactly one sensible reading, so there is nothing to clarify.
    assert repl._expand_indexes(["5-2"], 10, quiet) == [2, 3, 4, 5]


def test_a_range_of_one_is_just_that_index(quiet: Renderer) -> None:
    assert repl._expand_indexes(["4-4"], 10, quiet) == [4]


def test_duplicates_collapse_and_the_result_is_sorted(quiet: Renderer) -> None:
    # The list is shown back for confirmation, so it should read in the same
    # order as the numbering the user just saw.
    assert repl._expand_indexes(["3", "1-3", "1"], 10, quiet) == [1, 2, 3]
    assert repl._expand_indexes(["5", "2-3"], 10, quiet) == [2, 3, 5]


def test_a_range_past_the_end_is_refused_whole(quiet: Renderer) -> None:
    assert repl._expand_indexes(["1-99"], 5, quiet) is None
    assert "超出范围" in quiet.err.getvalue()


def test_a_range_starting_below_one_is_refused(quiet: Renderer) -> None:
    assert repl._expand_indexes(["0-2"], 5, quiet) is None


def test_half_a_range_is_refused(quiet: Renderer) -> None:
    for token in ("1-", "-3", "a-b", "1..3"):
        quiet.err.seek(0)
        quiet.err.truncate()
        assert repl._expand_indexes([token], 5, quiet) is None, token
        assert "看不懂" in quiet.err.getvalue(), token


def test_forget_accepts_a_range(app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(app_dir)
    for index in range(1, 6):
        write_log(app_dir, f"2026100{index}-000000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget 2-4")

    remaining = sorted(item.name for item in (app_dir / "history").glob("*.jsonl"))
    assert remaining == ["20261001-000000.jsonl", "20261005-000000.jsonl"]


def test_forget_refuses_a_range_that_runs_past_the_end(
    app_dir: Path, console: Renderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(app_dir)
    write_log(app_dir, "20261001-000000.jsonl", *a_session())
    approve(monkeypatch, console)

    repl._handle_command(engine, console, "/forget 1-9")

    assert (app_dir / "history" / "20261001-000000.jsonl").exists()
    assert "超出范围" in console.err.getvalue()
