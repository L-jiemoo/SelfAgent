"""Project context: the conventions of the directory you are working in.

Skills travel on the USB stick and say how *you* like to work. This is the other
half -- what *this project* expects -- read from the workspace root and folded
into the system prompt, the way a CLAUDE.md is.

Kept separate from skills on purpose. A skill is a capability you carry between
machines; this is a property of one directory, and it should be checked in
alongside the code it describes.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_FILE = "AGENTS.md"
MAX_CHARS = 8000


def load(
    root: Path,
    name: str = DEFAULT_FILE,
    max_chars: int = MAX_CHARS,
) -> tuple[str, Path | None]:
    """Read the context file. Returns (prompt block, path).

    The block is "" when there is nothing to add, so callers can just append it.
    """
    if not name:
        return "", None

    candidate = Path(root) / name
    if not candidate.is_file():
        return "", None

    try:
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "", None
    if not text:
        return "", None

    if len(text) > max_chars:
        # Truncated rather than refused: half a project brief still beats none,
        # and the marker tells the model that something is missing.
        text = text[:max_chars].rstrip() + "\n\n(内容过长,已截断)"

    block = (
        f"\n\nProject context, read from {candidate}:\n"
        "These are the conventions for the directory you are working in; follow "
        "them unless the user says otherwise.\n\n"
        f"{text}"
    )
    return block, candidate
