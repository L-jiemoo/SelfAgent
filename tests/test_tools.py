"""Sandbox and tool behaviour.

The sandbox tests are the important ones. Everything else here is convenience;
path confinement is the only thing standing between a prompt injected into some
file the model reads and arbitrary writes on a stranger's machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from selfagent.providers.base import ToolCall
from selfagent.tools import Toolbox


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    secret = tmp_path / "outside.txt"
    secret.write_text("not yours\n", encoding="utf-8")
    return secret


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "hello.txt").write_text("line one\nline two\nline three\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("nested content\n", encoding="utf-8")
    return root


class Confirm:
    """Records every approval request so tests can assert on the diff shown."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.calls: list[tuple] = []

    def __call__(self, action, path, old_text, new_text) -> bool:
        self.calls.append((action, path, old_text, new_text))
        return self.approve


def make(workspace: Path, *, allow_write: bool = True, approve: bool = True, read_roots=()):
    confirm = Confirm(approve)
    box = Toolbox(
        root=workspace,
        confirm=confirm,
        allow_write=allow_write,
        read_roots=tuple(read_roots),
    )
    return box, confirm


def run(box: Toolbox, name: str, **args):
    return box.execute(ToolCall(id="call_1", name=name, arguments=args))


# ---- sandbox ---------------------------------------------------------


def test_reads_a_file_inside_the_workspace(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", path="hello.txt")
    assert not result.is_error
    assert "line one" in result.content


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", r"..\..\outside.txt", "sub/../../outside.txt"],
)
def test_dot_dot_escape_is_refused(workspace: Path, path: str) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", path=path)
    assert result.is_error
    assert "越界" in result.content


def test_absolute_path_outside_is_refused(workspace: Path, outside: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", path=str(outside))
    assert result.is_error
    assert "越界" in result.content


def test_write_outside_is_refused_before_any_confirmation(workspace: Path, outside: Path) -> None:
    box, confirm = make(workspace)
    result = run(box, "write_file", path=str(outside), content="pwned")
    assert result.is_error
    assert confirm.calls == [], "should never reach the approval prompt"
    assert outside.read_text(encoding="utf-8") == "not yours\n"


def test_symlink_pointing_outside_is_refused(workspace: Path, outside: Path) -> None:
    link = workspace / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable (needs developer mode or admin)")

    box, _ = make(workspace)
    result = run(box, "read_file", path="link.txt")
    assert result.is_error
    assert "越界" in result.content


def test_the_workspace_root_itself_is_allowed(workspace: Path) -> None:
    box, _ = make(workspace)
    assert not run(box, "list_dir", path=".").is_error


def test_empty_path_is_refused(workspace: Path) -> None:
    box, _ = make(workspace)
    assert run(box, "read_file", path="  ").is_error


# ---- read-only roots (installed skills) ------------------------------
#
# A skill folder ships beside the program while the workspace is wherever the
# user launched from, so the two are different trees. The exemption is read-only
# on purpose: it must not become a write primitive.


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    (root / "greeter").mkdir(parents=True)
    (root / "greeter" / "SKILL.md").write_text("# greeter\n", encoding="utf-8")
    return root


def test_a_read_root_can_be_read(workspace: Path, skills_root: Path) -> None:
    box, _ = make(workspace, read_roots=(skills_root,))
    result = run(box, "read_file", path=str(skills_root / "greeter" / "SKILL.md"))
    assert not result.is_error
    assert "greeter" in result.content


def test_a_read_root_is_labelled_against_its_parent(workspace: Path, skills_root: Path) -> None:
    box, _ = make(workspace, read_roots=(skills_root,))
    result = run(box, "read_file", path=str(skills_root / "greeter" / "SKILL.md"))
    expected = str(Path("skills") / "greeter" / "SKILL.md")
    assert expected in result.content


def test_a_read_root_can_be_listed_and_searched(workspace: Path, skills_root: Path) -> None:
    box, _ = make(workspace, read_roots=(skills_root,))
    assert not run(box, "list_dir", path=str(skills_root)).is_error
    found = run(box, "search", pattern="greeter", path=str(skills_root))
    assert not found.is_error
    assert "SKILL.md" in found.content


def test_writing_into_a_read_root_is_refused(workspace: Path, skills_root: Path) -> None:
    box, confirm = make(workspace, read_roots=(skills_root,))
    target = skills_root / "greeter" / "SKILL.md"

    result = run(box, "write_file", path=str(target), content="pwned")

    assert result.is_error
    assert confirm.calls == [], "should never reach the approval prompt"
    assert target.read_text(encoding="utf-8") == "# greeter\n"


def test_creating_a_file_in_a_read_root_is_refused(workspace: Path, skills_root: Path) -> None:
    box, confirm = make(workspace, read_roots=(skills_root,))
    result = run(box, "write_file", path=str(skills_root / "new.md"), content="x")

    assert result.is_error
    assert confirm.calls == []
    assert not (skills_root / "new.md").exists()


def test_editing_inside_a_read_root_is_refused(workspace: Path, skills_root: Path) -> None:
    box, confirm = make(workspace, read_roots=(skills_root,))
    result = run(
        box,
        "edit_file",
        path=str(skills_root / "greeter" / "SKILL.md"),
        old_string="greeter",
        new_string="x",
    )

    assert result.is_error
    assert confirm.calls == []
    assert (skills_root / "greeter" / "SKILL.md").read_text(encoding="utf-8") == "# greeter\n"


def test_a_read_root_does_not_widen_access_elsewhere(
    workspace: Path, skills_root: Path, outside: Path
) -> None:
    box, _ = make(workspace, read_roots=(skills_root,))
    assert run(box, "read_file", path=str(outside)).is_error


def test_dot_dot_out_of_a_read_root_is_refused(
    workspace: Path, skills_root: Path, outside: Path
) -> None:
    # skills/../outside.txt is a sibling of the read root, not inside it.
    box, _ = make(workspace, read_roots=(skills_root,))
    escape = skills_root / ".." / "outside.txt"
    assert run(box, "read_file", path=str(escape)).is_error


# ---- read tools ------------------------------------------------------


def test_read_missing_file(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", path="nope.txt")
    assert result.is_error
    assert "不存在" in result.content


def test_read_directory_suggests_list_dir(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", path="sub")
    assert result.is_error
    assert "list_dir" in result.content


def test_read_pages_and_says_how_much_is_left(workspace: Path) -> None:
    (workspace / "big.txt").write_text(
        "\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8"
    )
    box, _ = make(workspace)

    first = run(box, "read_file", path="big.txt", offset=1, limit=5)
    assert "line 1" in first.content
    assert "line 6" not in first.content
    assert "还有" in first.content

    second = run(box, "read_file", path="big.txt", offset=6, limit=5)
    assert "line 6" in second.content


def test_list_dir_marks_directories(workspace: Path) -> None:
    box, _ = make(workspace)
    content = run(box, "list_dir", path=".").content
    assert "sub/" in content
    assert "hello.txt" in content


def test_search_finds_matches_with_line_numbers(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "search", pattern="nested")
    assert not result.is_error
    assert "nested.txt:1" in result.content


def test_search_reports_no_matches_without_erroring(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "search", pattern="zzzz-not-here")
    assert not result.is_error
    assert "没有匹配" in result.content


def test_search_rejects_a_bad_regex(workspace: Path) -> None:
    box, _ = make(workspace)
    assert run(box, "search", pattern="[unclosed").is_error


# ---- edit_file -------------------------------------------------------


def test_edit_replaces_a_unique_string(workspace: Path) -> None:
    box, confirm = make(workspace)
    result = run(box, "edit_file", path="hello.txt", old_string="line two", new_string="line 2")

    assert not result.is_error
    assert (workspace / "hello.txt").read_text(encoding="utf-8").startswith("line one\nline 2\n")
    assert len(confirm.calls) == 1
    action, path, old, new = confirm.calls[0]
    assert "line two" in old and "line 2" in new


def test_edit_refuses_an_ambiguous_match(workspace: Path) -> None:
    (workspace / "dup.txt").write_text("aaa\naaa\n", encoding="utf-8")
    box, confirm = make(workspace)

    result = run(box, "edit_file", path="dup.txt", old_string="aaa", new_string="bbb")
    assert result.is_error
    assert "出现 2 次" in result.content
    assert confirm.calls == []
    assert (workspace / "dup.txt").read_text(encoding="utf-8") == "aaa\naaa\n"


def test_edit_replace_all_when_asked(workspace: Path) -> None:
    (workspace / "dup.txt").write_text("aaa\naaa\n", encoding="utf-8")
    box, _ = make(workspace)
    result = run(box, "edit_file", path="dup.txt", old_string="aaa", new_string="b", replace_all=True)
    assert not result.is_error
    assert (workspace / "dup.txt").read_text(encoding="utf-8") == "b\nb\n"


def test_edit_missing_string_does_not_touch_the_file(workspace: Path) -> None:
    box, confirm = make(workspace)
    result = run(box, "edit_file", path="hello.txt", old_string="absent", new_string="x")
    assert result.is_error
    assert confirm.calls == []
    assert "line one" in (workspace / "hello.txt").read_text(encoding="utf-8")


def test_edit_missing_file(workspace: Path) -> None:
    box, _ = make(workspace)
    assert run(box, "edit_file", path="ghost.txt", old_string="a", new_string="b").is_error


# ---- approval --------------------------------------------------------


def test_refused_write_leaves_no_trace_and_is_not_a_crash(workspace: Path) -> None:
    box, confirm = make(workspace, approve=False)
    result = run(box, "write_file", path="new.txt", content="hello")

    # Must come back as an ordinary error result -- the model has to be told,
    # and on Anthropic every call needs an answer or the next request 400s.
    assert result.is_error
    assert result.id == "call_1"
    assert "用户拒绝" in result.content
    assert not (workspace / "new.txt").exists()
    assert len(confirm.calls) == 1


def test_write_creates_a_new_file_when_approved(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "write_file", path="made/up.txt", content="fresh")
    assert not result.is_error
    assert (workspace / "made" / "up.txt").read_text(encoding="utf-8") == "fresh"


def test_read_only_mode_refuses_writes_without_prompting(workspace: Path) -> None:
    box, confirm = make(workspace, allow_write=False)
    result = run(box, "write_file", path="x.txt", content="y")

    assert result.is_error
    assert "--allow-write" in result.content
    assert confirm.calls == []
    assert not (workspace / "x.txt").exists()


def test_read_only_mode_still_allows_reading(workspace: Path) -> None:
    box, _ = make(workspace, allow_write=False)
    assert not run(box, "read_file", path="hello.txt").is_error


# ---- dispatch --------------------------------------------------------


def test_unknown_tool_is_reported_not_raised(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "delete_everything", path=".")
    assert result.is_error
    assert "未知工具" in result.content


def test_bad_arguments_are_reported_not_raised(workspace: Path) -> None:
    box, _ = make(workspace)
    result = run(box, "read_file", wrong_kwarg="x")
    assert result.is_error
    assert "参数不正确" in result.content


def test_specs_cover_every_dispatched_handler(workspace: Path) -> None:
    box, _ = make(workspace)
    declared = {spec.name for spec in box.specs()}
    assert declared == {"read_file", "list_dir", "search", "write_file", "edit_file"}
