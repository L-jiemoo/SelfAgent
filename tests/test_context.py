"""Reading a project's own conventions out of the workspace.

The counterpart to skills: a skill travels on the USB stick and says how you
like to work, while this lives with the code and says what the project expects.
"""

from __future__ import annotations

from pathlib import Path

from selfagent import context


def test_a_missing_file_adds_nothing(tmp_path: Path) -> None:
    assert context.load(tmp_path) == ("", None)


def test_an_empty_file_adds_nothing(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n\n", encoding="utf-8")
    assert context.load(tmp_path) == ("", None)


def test_the_file_becomes_a_prompt_block(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Always use tabs.", encoding="utf-8")

    block, path = context.load(tmp_path)

    assert "Always use tabs." in block
    assert path == tmp_path / "AGENTS.md"
    assert str(path) in block, "the model should be able to say where this came from"


def test_a_custom_name_can_be_used(tmp_path: Path) -> None:
    (tmp_path / "CONVENTIONS.md").write_text("Use two spaces.", encoding="utf-8")
    block, _ = context.load(tmp_path, "CONVENTIONS.md")
    assert "Use two spaces." in block


def test_an_empty_name_disables_it(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x", encoding="utf-8")
    assert context.load(tmp_path, "") == ("", None)


def test_a_nested_name_is_allowed(tmp_path: Path) -> None:
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "CONTEXT.md").write_text("Nested.", encoding="utf-8")

    block, path = context.load(tmp_path, ".agent/CONTEXT.md")

    assert "Nested." in block
    assert path is not None and path.name == "CONTEXT.md"


def test_a_long_file_is_truncated_with_a_marker(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 5000, encoding="utf-8")

    block, _ = context.load(tmp_path, max_chars=100)

    assert "已截断" in block
    assert len(block) < 1000, "a project brief must not be allowed to eat the context"


def test_a_directory_named_like_the_file_is_not_a_file(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").mkdir()
    assert context.load(tmp_path) == ("", None)
