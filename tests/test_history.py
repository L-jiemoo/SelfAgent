"""Reading the conversation log back in.

Resume is only as good as its reader, and the reader has to survive logs it did
not write: ones cut off mid-line by a pulled USB stick, and older ones that
recorded tool calls without ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from selfagent import history
from selfagent.config import load_config
from selfagent.providers.base import Message, ToolCall, ToolResult
from selfagent.session import COMPACT_MARKER, Session


def write_log(directory: Path, name: str, *records: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def started(target: str = "deepseek/deepseek-v4-flash", when: str = "2026-10-16T15:30:00"):
    return {"type": "session_start", "time": when, "target": target}


def one_call(name: str = "read_file") -> dict[str, Any]:
    return {"type": "assistant", "text": "", "tool_calls": [{"name": name, "arguments": {}}]}


# ---- listing ---------------------------------------------------------


def test_sessions_are_listed_newest_first(tmp_path: Path) -> None:
    write_log(tmp_path, "20260101-090000.jsonl", started(), {"type": "user", "text": "旧的"})
    write_log(tmp_path, "20261016-153000.jsonl", started(), {"type": "user", "text": "新的"})

    names = [item.path.name for item in history.list_sessions(tmp_path)]
    assert names == ["20261016-153000.jsonl", "20260101-090000.jsonl"]


def test_a_foreign_jsonl_is_not_a_session(tmp_path: Path) -> None:
    (tmp_path / "not-ours.jsonl").write_text('{"hello": "world"}\n', encoding="utf-8")
    assert history.list_sessions(tmp_path) == []


def test_non_jsonl_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    assert history.list_sessions(tmp_path) == []


def test_a_missing_directory_lists_nothing(tmp_path: Path) -> None:
    assert history.list_sessions(tmp_path / "nope") == []


def test_the_limit_is_respected(tmp_path: Path) -> None:
    for index in range(5):
        write_log(tmp_path, f"2026010{index}-000000.jsonl", started(), {"type": "user", "text": "q"})
    assert len(history.list_sessions(tmp_path, limit=3)) == 3


def test_the_limit_can_be_removed_for_cleanup(tmp_path: Path) -> None:
    # Cleanup has to see everything it might delete, not just the recent ones.
    for index in range(5):
        write_log(tmp_path, f"2026010{index}-000000.jsonl", started(), {"type": "user", "text": "q"})
    assert len(history.list_sessions(tmp_path, limit=None)) == 5


# ---- removing --------------------------------------------------------


def test_removing_a_log_reports_success(tmp_path: Path) -> None:
    path = write_log(tmp_path, "a.jsonl", started(), {"type": "user", "text": "q"})
    assert history.size_of(path) > 0
    assert history.remove(path) is True
    assert not path.exists()


def test_removing_an_already_missing_log_counts_as_done(tmp_path: Path) -> None:
    assert history.remove(tmp_path / "ghost.jsonl") is True


def test_size_of_a_missing_file_is_zero(tmp_path: Path) -> None:
    assert history.size_of(tmp_path / "ghost.jsonl") == 0


# ---- summaries -------------------------------------------------------


def test_summary_counts_turns_and_previews_the_first_question(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "20261016-153000.jsonl",
        started(),
        {"type": "user", "text": "帮我看看这个项目的结构"},
        {"type": "assistant", "text": "好的"},
        {"type": "user", "text": "再详细一点"},
        {"type": "assistant", "text": "这样"},
    )

    summary = history.summarise(path)

    assert summary is not None
    assert summary.turns == 2
    assert summary.target == "deepseek/deepseek-v4-flash"
    assert summary.preview == "帮我看看这个项目的结构"
    assert summary.started == "2026-10-16T15:30:00"


def test_a_tool_round_trip_is_not_a_turn(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "a.jsonl",
        started(),
        {"type": "user", "text": "q"},
        one_call(),
        {"type": "tool_results", "results": [{"content": "x"}]},
        {"type": "assistant", "text": "答案"},
    )
    summary = history.summarise(path)
    assert summary is not None
    assert summary.turns == 1


def test_the_preview_is_truncated(tmp_path: Path) -> None:
    path = write_log(tmp_path, "a.jsonl", started(), {"type": "user", "text": "字" * 200})
    summary = history.summarise(path)
    assert summary is not None
    assert len(summary.preview) == history.PREVIEW_CHARS
    assert summary.preview.endswith("…")


def test_a_foreign_file_has_no_summary(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"type": "whatever"}\n', encoding="utf-8")
    assert history.summarise(path) is None


# ---- loading ---------------------------------------------------------


def test_plain_turns_round_trip(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "a.jsonl",
        started(),
        {"type": "user", "text": "问题"},
        {"type": "assistant", "text": "回答"},
    )
    messages = history.load(path)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "问题"
    assert messages[1].content == "回答"


def test_old_logs_without_ids_still_pair_their_tool_calls(tmp_path: Path) -> None:
    # The pre-id log format. Without a matching id the API rejects the tool_use
    # outright, so pairing positionally is what makes these logs resumable.
    path = write_log(
        tmp_path,
        "old.jsonl",
        started(),
        {"type": "user", "text": "读一下 a.txt"},
        one_call(),
        {"type": "tool_results", "results": [{"name": "", "is_error": False, "content": "内容"}]},
        {"type": "assistant", "text": "读到了"},
    )

    messages = history.load(path)
    call = messages[1].tool_calls[0]
    result = messages[2].tool_results[0]

    assert call.name == "read_file"
    assert call.id == result.id


def test_recorded_ids_are_kept(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "new.jsonl",
        started(),
        {"type": "user", "text": "q"},
        {
            "type": "assistant",
            "text": "",
            "tool_calls": [{"id": "call_abc", "name": "search", "arguments": {}}],
        },
        {"type": "tool_results", "results": [{"id": "call_abc", "is_error": True, "content": "没了"}]},
    )

    messages = history.load(path)
    assert messages[1].tool_calls[0].id == "call_abc"
    assert messages[2].tool_results[0].id == "call_abc"
    assert messages[2].tool_results[0].is_error is True


def test_parallel_calls_keep_their_own_pairs(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "par.jsonl",
        started(),
        {"type": "user", "text": "q"},
        {
            "type": "assistant",
            "text": "",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "a"}},
                           {"name": "read_file", "arguments": {"path": "b"}}],
        },
        {"type": "tool_results", "results": [{"content": "A"}, {"content": "B"}]},
    )

    messages = history.load(path)
    calls = messages[1].tool_calls
    results = messages[2].tool_results

    assert [call.id for call in calls] == [result.id for result in results]
    assert len({call.id for call in calls}) == 2


def test_a_half_written_last_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "cut.jsonl"
    path.write_text(
        json.dumps(started(), ensure_ascii=False) + "\n"
        + json.dumps({"type": "user", "text": "问题"}, ensure_ascii=False) + "\n"
        + '{"type": "assistant", "te',
        encoding="utf-8",
    )
    assert [message.content for message in history.load(path)] == ["问题"]


def test_an_unanswered_tool_call_is_closed_off(tmp_path: Path) -> None:
    # The log ends between the assistant's call and its results.
    path = write_log(tmp_path, "cut.jsonl", started(), {"type": "user", "text": "q"}, one_call())

    messages = history.load(path)

    assert messages[-1].role == "user"
    assert messages[-1].tool_results[0].id == messages[-2].tool_calls[0].id
    assert messages[-1].tool_results[0].is_error is True


def test_a_log_opening_with_an_assistant_turn_is_repaired(tmp_path: Path) -> None:
    path = write_log(tmp_path, "odd.jsonl", started(), {"type": "assistant", "text": "半路开始"})
    assert history.load(path)[0].role == "user"


def test_a_compact_record_resets_the_conversation(tmp_path: Path) -> None:
    path = write_log(
        tmp_path,
        "c.jsonl",
        started(),
        {"type": "user", "text": "很久以前"},
        {"type": "assistant", "text": "是的"},
        {"type": "compact", "summary": "摘要"},
        {"type": "user", "text": "压缩之后"},
        {"type": "assistant", "text": "嗯"},
    )

    messages = history.load(path)

    assert messages[0].content == COMPACT_MARKER
    assert messages[1].content == "摘要"
    assert [message.content for message in messages[2:]] == ["压缩之后", "嗯"]


def test_an_empty_log_loads_nothing(tmp_path: Path) -> None:
    assert history.load(write_log(tmp_path, "e.jsonl", started())) == []


def test_a_missing_file_loads_nothing(tmp_path: Path) -> None:
    assert history.load(tmp_path / "ghost.jsonl") == []


# ---- adopting it -----------------------------------------------------


def test_restore_recomputes_the_turn_counter(app_dir: Path) -> None:
    session = Session(load_config(), "deepseek/x", persist=False)
    session.restore(
        [
            Message("user", "q"),
            Message("assistant", "a"),
            Message("user", "q2"),
            Message("assistant", "", tool_calls=[ToolCall("c1", "read_file", {})]),
            Message("user", "", tool_results=[ToolResult("c1", "ok")]),
            Message("assistant", "a2"),
        ]
    )

    assert len(session.messages) == 6
    assert session.turns == 2, "a tool round trip is not a finished turn"
