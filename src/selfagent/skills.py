"""Skills: instruction packs that ride along on the USB stick.

A skill is a folder holding a SKILL.md. Only the frontmatter -- name and
description -- is read at startup and folded into the system prompt; the body is
fetched by the model with `read_file` when it decides the skill is relevant.
That split is the whole point: the catalogue costs a couple of lines per
request, the body costs nothing until it is actually needed.

Two things a skill deliberately cannot do, because this program runs off a USB
stick on machines that are not the owner's:

* Execute anything. There is no shell tool, so a skill that ships a script
  ships *documentation about* a script, not a capability.
* Reach the network. Nothing here downloads skills; installing one means
  copying a folder into skills/.

Both restrictions are the same ones the file tools already live under, which is
why skills need no new machinery -- just a read-only root and a prompt block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SKILL_FILE = "SKILL.md"
MAX_DESCRIPTION = 300
NO_DESCRIPTION = "(no description)"

# A leading `---` block, up to the closing `---` on its own line.
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path  # absolute path to the SKILL.md itself

    @property
    def directory(self) -> Path:
        return self.path.parent


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return the `---` block as flat key/value pairs, plus the remaining body.

    Hand-rolled rather than pulling in PyYAML. The only fields that matter are
    flat `key: value` lines, and a new dependency has to earn its place in a
    bundle that ships on a USB stick and is unpacked on every launch.
    """
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text

    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        value = value.strip().strip("'\"")
        if value:
            fields[key.strip().lower()] = value
    return fields, text[match.end() :]


def _first_paragraph(body: str) -> str:
    """A description of last resort, for a SKILL.md that has no frontmatter.

    Headings are skipped while anything else remains: a lone `# Title` says
    nothing about when to reach for the skill, whereas the paragraph under it
    usually does. A heading-only file still falls back to its heading.
    """
    blocks = [
        " ".join(part.strip() for part in block.splitlines() if part.strip())
        for block in re.split(r"\r?\n[ \t]*\r?\n", body.strip())
    ]
    blocks = [block for block in blocks if block]
    for block in blocks:
        if not block.lstrip().startswith("#"):
            return block.strip()
    return blocks[0].lstrip("#").strip() if blocks else ""


def _clip(text: str) -> str:
    if len(text) <= MAX_DESCRIPTION:
        return text
    return text[: MAX_DESCRIPTION - 1].rstrip() + "…"


def _load(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    fields, body = _split_frontmatter(text)
    name = fields.get("name") or path.parent.name
    description = fields.get("description") or _first_paragraph(body)
    return Skill(name=name, description=_clip(description) or NO_DESCRIPTION, path=path)


def discover(directory: Path) -> list[Skill]:
    """Every skill directly under `directory`, sorted by name.

    Only immediate subdirectories are considered, so an asset tree inside a
    skill cannot masquerade as a skill of its own.
    """
    if not directory.is_dir():
        return []

    try:
        children = sorted(directory.iterdir())
    except OSError:
        return []

    found: list[Skill] = []
    for child in children:
        if not child.is_dir():
            continue
        candidate = child / SKILL_FILE
        if not candidate.is_file():
            continue
        skill = _load(candidate)
        if skill is not None:
            found.append(skill)
    return sorted(found, key=lambda item: item.name.lower())


def catalog(skills: list[Skill], directory: Path) -> str:
    """The block appended to the system prompt.

    Deliberately terse -- it is paid for on every single request. It names each
    skill, gives the absolute path to open, and says not to preload.
    """
    if not skills:
        return ""

    lines = [
        "",
        "",
        "Available skills. Read a SKILL.md with read_file only when it genuinely",
        "matches the task, then follow it; do not preload all of them.",
    ]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}  [{skill.path}]")
    lines.append(f"Skills are read-only and live under: {directory}")
    return "\n".join(lines)
