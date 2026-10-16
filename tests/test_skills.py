"""Skill discovery, and the catalogue that reaches the system prompt.

Two properties matter here and both are load-bearing. The catalogue must name
every skill and say where to open it -- otherwise the model never finds one. And
it must *not* carry the bodies -- otherwise every request pays for every
installed skill, which is exactly the cost the split exists to avoid.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from selfagent import skills
from selfagent.skills import catalog, discover


def write_skill(root: Path, folder: str, text: str) -> Path:
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


# ---- frontmatter -----------------------------------------------------


def test_reads_name_and_description(tmp_path: Path) -> None:
    write_skill(tmp_path, "greeter", "---\nname: greeter\ndescription: Say hello\n---\n\nBody.\n")
    found = discover(tmp_path)
    assert len(found) == 1
    assert found[0].name == "greeter"
    assert found[0].description == "Say hello"


def test_folder_name_is_the_fallback_name(tmp_path: Path) -> None:
    write_skill(tmp_path, "nameless", "---\ndescription: No name given\n---\n\nx\n")
    assert discover(tmp_path)[0].name == "nameless"


def test_first_paragraph_is_the_fallback_description(tmp_path: Path) -> None:
    write_skill(tmp_path, "plain", "# Plain\n\nUse this when the task is plain.\n\nMore.\n")
    skill = discover(tmp_path)[0]
    assert skill.name == "plain"
    assert skill.description == "Use this when the task is plain."


def test_a_heading_only_file_falls_back_to_the_heading(tmp_path: Path) -> None:
    write_skill(tmp_path, "heading", "# Heading Only\n")
    assert discover(tmp_path)[0].description == "Heading Only"


def test_empty_file_gets_a_placeholder_not_a_crash(tmp_path: Path) -> None:
    write_skill(tmp_path, "empty", "")
    assert discover(tmp_path)[0].description == skills.NO_DESCRIPTION


def test_long_description_is_clipped(tmp_path: Path) -> None:
    write_skill(tmp_path, "wordy", "---\nname: wordy\ndescription: " + "x" * 500 + "\n---\n")
    assert len(discover(tmp_path)[0].description) <= skills.MAX_DESCRIPTION


def test_quoted_values_are_unquoted(tmp_path: Path) -> None:
    write_skill(tmp_path, "q", "---\nname: \"quoted\"\ndescription: 'single'\n---\n")
    skill = discover(tmp_path)[0]
    assert skill.name == "quoted"
    assert skill.description == "single"


def test_crlf_frontmatter_is_understood(tmp_path: Path) -> None:
    # A SKILL.md written in Notepad on Windows arrives with CRLF line endings.
    write_skill(
        tmp_path, "crlf", "---\r\nname: crlf\r\ndescription: From Windows\r\n---\r\n\r\nBody\r\n"
    )
    skill = discover(tmp_path)[0]
    assert skill.name == "crlf"
    assert skill.description == "From Windows"


# ---- discovery -------------------------------------------------------


def test_nested_folders_are_not_skills(tmp_path: Path) -> None:
    write_skill(tmp_path, "outer", "---\nname: outer\n---\n")
    write_skill(tmp_path / "outer", "inner", "---\nname: inner\n---\n")
    assert [item.name for item in discover(tmp_path)] == ["outer"]


def test_folders_without_a_skill_file_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "README.md").write_text("hi", encoding="utf-8")
    (tmp_path / "loose.md").write_text("hi", encoding="utf-8")
    assert discover(tmp_path) == []


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    assert discover(tmp_path / "nope") == []


def test_skills_are_sorted_by_name(tmp_path: Path) -> None:
    write_skill(tmp_path, "zeta", "---\nname: zeta\n---\n")
    write_skill(tmp_path, "alpha", "---\nname: alpha\n---\n")
    assert [item.name for item in discover(tmp_path)] == ["alpha", "zeta"]


def test_skill_points_at_its_own_skill_md(tmp_path: Path) -> None:
    path = write_skill(tmp_path, "greeter", "---\nname: greeter\n---\n")
    found = discover(tmp_path)[0]
    assert found.path == path
    assert found.directory == tmp_path / "greeter"


# ---- catalogue -------------------------------------------------------


def test_catalogue_is_empty_without_skills(tmp_path: Path) -> None:
    assert catalog([], tmp_path) == ""


def test_catalogue_names_every_skill_and_where_to_open_it(tmp_path: Path) -> None:
    path = write_skill(tmp_path, "greeter", "---\nname: greeter\ndescription: Say hello\n---\n")
    text = catalog(discover(tmp_path), tmp_path)
    assert "greeter" in text
    assert "Say hello" in text
    assert str(path) in text, "the model needs a path it can hand to read_file"


def test_catalogue_does_not_embed_the_body(tmp_path: Path) -> None:
    write_skill(
        tmp_path, "greeter", "---\nname: greeter\ndescription: Say hello\n---\n\nBODY-MARKER\n"
    )
    assert "BODY-MARKER" not in catalog(discover(tmp_path), tmp_path)


def test_catalogue_tells_the_model_not_to_preload(tmp_path: Path) -> None:
    write_skill(tmp_path, "greeter", "---\nname: greeter\n---\n")
    assert "do not preload" in catalog(discover(tmp_path), tmp_path)


# ---- wiring ----------------------------------------------------------


class _Config:
    """A fresh stand-in each time -- a class attribute would leak between tests."""

    def __init__(self) -> None:
        self.generation = SimpleNamespace(system="You are a helpful assistant.")


def test_catalogue_is_appended_to_the_system_prompt(tmp_path: Path) -> None:
    from selfagent.cli import _inject_catalogue

    write_skill(tmp_path, "greeter", "---\nname: greeter\ndescription: Say hello\n---\n")
    config = _Config()
    _inject_catalogue(config, discover(tmp_path), tmp_path)

    assert config.generation.system.startswith("You are a helpful assistant.")
    assert "greeter" in config.generation.system


def test_injecting_nothing_leaves_the_prompt_alone(tmp_path: Path) -> None:
    from selfagent.cli import _inject_catalogue

    config = _Config()
    _inject_catalogue(config, [], tmp_path)
    assert config.generation.system == "You are a helpful assistant."


def test_skills_are_found_next_to_the_program(app_dir: Path) -> None:
    from selfagent.cli import _discover_skills
    from selfagent.config import load_config

    write_skill(app_dir / "skills", "greeter", "---\nname: greeter\ndescription: hi\n---\n")
    root, found = _discover_skills(load_config(), SimpleNamespace(no_skills=False))

    assert root == app_dir / "skills"
    assert [item.name for item in found] == ["greeter"]


def test_no_skills_switch_skips_discovery(app_dir: Path) -> None:
    from selfagent.cli import _discover_skills
    from selfagent.config import load_config

    write_skill(app_dir / "skills", "greeter", "---\nname: greeter\n---\n")
    root, found = _discover_skills(load_config(), SimpleNamespace(no_skills=True))

    assert root == Path()
    assert found == []


def test_skills_off_in_config_skips_discovery(app_dir: Path) -> None:
    from selfagent.cli import _discover_skills
    from selfagent.config import load_config

    write_skill(app_dir / "skills", "greeter", "---\nname: greeter\n---\n")
    config = load_config()
    config.skills.enabled = False
    assert _discover_skills(config, SimpleNamespace(no_skills=False)) == (Path(), [])
