"""File tools, confined to a single workspace root.

The sandbox is the point of this module, not a detail of it. This program runs
off a USB stick on machines that are not the owner's, and the model driving it
reads files whose contents it does not control. A prompt injected into some file
it reads could try to make it write elsewhere; the only thing standing between
that and arbitrary file writes is `Toolbox._resolve`.

Everything is resolved before it is checked, so `..` traversal and symlinks
pointing outside the root are both caught -- a textual prefix check on the raw
argument would miss both.

Tools never prompt on their own. Writes go through a `confirm` callback supplied
by the CLI layer, which keeps this module testable and keeps the terminal
concerns out of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import ToolError
from .providers.base import ToolCall, ToolResult, ToolSpec

MAX_READ_BYTES = 256 * 1024
MAX_LIST_ENTRIES = 500
MAX_SEARCH_MATCHES = 200
DEFAULT_READ_LINES = 2000
# How many writes can be taken back. Bounded because each record holds the
# file's previous contents in memory.
MAX_UNDO = 20

# Files that are almost never what anyone means, and that would waste the
# model's context on noise.
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".idea", ".mypy_cache"}

# Not a security boundary -- the sandbox is. But "the agent tidied my code" and
# "the agent rotated my credentials" are different kinds of event, and the second
# one is worth a hard stop even inside an authorised workspace.
PROTECTED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.enc",
}
PROTECTED_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".keystore", ".jks")
PROTECTED_DIRS = {".git", ".ssh", ".aws", ".gnupg"}

# (action label, path, old text, new text) -> approved
ConfirmFn = Callable[[str, Path, str, str], bool]


def _protected_reason(path: Path) -> str:
    name = path.name.lower()
    if name in PROTECTED_NAMES:
        return "凭据类文件"
    if name.endswith(PROTECTED_SUFFIXES):
        return "密钥或证书"
    for part in path.parts:
        if part.lower() in PROTECTED_DIRS:
            return f"{part}/ 是版本控制或密钥目录"
    return ""


@dataclass
class WriteRecord:
    """What a write replaced, so it can be put back."""

    path: Path
    action: str
    existed: bool
    previous: str
    written: str


@dataclass
class Toolbox:
    root: Path
    confirm: ConfirmFn
    allow_write: bool = True
    max_read_bytes: int = MAX_READ_BYTES
    # Extra trees that may be *read* but never written. Installed skills live
    # here: they ship beside the program, while the workspace is wherever the
    # user launched from, so the two are different trees by design. Keeping the
    # exemption read-only is what stops a skill -- or a prompt injected into one
    # -- from becoming a write primitive.
    read_roots: tuple[Path, ...] = ()
    # Refuse to rewrite credential-shaped paths even inside the workspace.
    protect: bool = True
    # Writes made this session, newest last, so they can be reviewed and undone.
    writes: list[WriteRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.read_roots = tuple(Path(extra).resolve() for extra in self.read_roots)

    # ---- sandbox ------------------------------------------------------

    @staticmethod
    def _inside(base: Path, candidate: Path) -> bool:
        return candidate == base or base in candidate.parents

    def _resolve(self, raw: str, *, writable: bool = False) -> Path:
        if not raw or not str(raw).strip():
            raise ToolError("路径不能为空。")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ToolError(f"路径无法解析:{raw}({exc.strerror or exc})") from None

        if self._inside(self.root, resolved):
            return resolved
        if not writable and any(self._inside(extra, resolved) for extra in self.read_roots):
            return resolved

        if writable:
            raise ToolError(
                f"路径越界,拒绝写入:{raw}。写入只能发生在工作目录 {self.root} 之内。"
            )
        allowed = ", ".join(str(extra) for extra in (self.root, *self.read_roots))
        raise ToolError(f"路径越界,拒绝访问:{raw}。只允许访问:{allowed}。")

    def label(self, path: Path) -> str:
        """Public name for a path, for callers that only need to print it."""
        return self._relative(Path(path))

    def _relative(self, path: Path) -> str:
        """A readable label for a path, whichever tree it came from.

        A skill resolves against its own parent so it shows as
        `skills/<name>/SKILL.md` rather than an absolute path with a drive
        letter that changes from machine to machine.
        """
        if self._inside(self.root, path):
            try:
                return str(path.relative_to(self.root)) or "."
            except ValueError:  # pragma: no cover - defensive
                pass
        for extra in self.read_roots:
            if self._inside(extra, path):
                try:
                    return str(path.relative_to(extra.parent)) or "."
                except ValueError:  # pragma: no cover - defensive
                    pass
        return str(path)

    # ---- dispatch -----------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="read_file",
                description=(
                    "Read a UTF-8 text file from the workspace, or an installed "
                    "skill. Returns the file content with a header giving the path "
                    "and line range. Use `offset` and `limit` to page through large "
                    "files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the workspace root."},
                        "offset": {"type": "integer", "description": "1-based line to start at."},
                        "limit": {"type": "integer", "description": "Maximum number of lines to return."},
                    },
                    "required": ["path"],
                },
            ),
            ToolSpec(
                name="list_dir",
                description="List the entries of a directory in the workspace.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path. Defaults to the workspace root."},
                    },
                },
            ),
            ToolSpec(
                name="search",
                description=(
                    "Search file contents by regular expression across the "
                    "workspace. Returns matching lines as path:line:text. "
                    "Use this to locate code instead of reading whole files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Python regular expression."},
                        "path": {"type": "string", "description": "Directory or file to search. Defaults to the workspace root."},
                        "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
                    },
                    "required": ["pattern"],
                },
            ),
            ToolSpec(
                name="write_file",
                description=(
                    "Create a file or replace its entire contents. The user is "
                    "asked to approve every write. Prefer edit_file for changing "
                    "part of an existing file."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the workspace root."},
                        "content": {"type": "string", "description": "Full new contents of the file."},
                    },
                    "required": ["path", "content"],
                },
            ),
            ToolSpec(
                name="edit_file",
                description=(
                    "Replace an exact string in a file. `old_string` must match "
                    "exactly once unless `replace_all` is true; if it matches "
                    "more than once the edit is rejected so you can add context. "
                    "The user is asked to approve every edit."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path, relative to the workspace root."},
                        "old_string": {"type": "string", "description": "Exact text to replace."},
                        "new_string": {"type": "string", "description": "Replacement text."},
                        "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            ),
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call. Failures come back as error results, never as raises.

        The model must still receive a result for every call it made -- a
        missing tool_result breaks the next request outright on Anthropic, and
        leaves the model guessing on OpenAI.
        """
        handler = {
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "search": self._search,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
        }.get(call.name)

        if handler is None:
            return ToolResult(call.id, f"未知工具 '{call.name}'。", is_error=True)

        try:
            return ToolResult(call.id, handler(**call.arguments))
        except ToolError as exc:
            return ToolResult(call.id, str(exc), is_error=True)
        except TypeError as exc:
            return ToolResult(call.id, f"参数不正确:{exc}", is_error=True)
        except OSError as exc:
            return ToolResult(call.id, f"文件操作失败:{exc.strerror or exc}", is_error=True)

    # ---- read-only tools ----------------------------------------------

    def _read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        target = self._resolve(path)
        if target.is_dir():
            raise ToolError(f"{self._relative(target)} 是目录,请用 list_dir。")
        if not target.exists():
            raise ToolError(f"文件不存在:{self._relative(target)}")

        raw = target.read_bytes()
        clipped = len(raw) > self.max_read_bytes
        text = raw[: self.max_read_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines()

        start = max(0, (offset or 1) - 1)
        count = limit if limit and limit > 0 else DEFAULT_READ_LINES
        window = lines[start : start + count]

        header = f"# {self._relative(target)} 行 {start + 1}-{start + len(window)} / 共 {len(lines)} 行"
        if clipped:
            header += f"(文件过大,只读取了前 {self.max_read_bytes} 字节)"
        body = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(window))

        if start + len(window) < len(lines):
            body += f"\n… 还有 {len(lines) - start - len(window)} 行,用 offset 继续读取"
        return f"{header}\n{body}" if body else f"{header}\n(空文件)"

    def _list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"目录不存在:{self._relative(target)}")
        if not target.is_dir():
            raise ToolError(f"{self._relative(target)} 不是目录,请用 read_file。")

        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = []
        for entry in entries[:MAX_LIST_ENTRIES]:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                try:
                    lines.append(f"{entry.name}\t{entry.stat().st_size} B")
                except OSError:
                    lines.append(entry.name)
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"… 共 {len(entries)} 项,只列出前 {MAX_LIST_ENTRIES} 项")
        return "\n".join(lines) if lines else "(空目录)"

    def _search(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        target = self._resolve(path)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"正则表达式无效:{exc}") from None

        candidates = [target] if target.is_file() else list(self._walk(target))
        if glob:
            candidates = [p for p in candidates if _matches_glob(p.name, glob)]

        hits: list[str] = []
        for file in candidates:
            if len(hits) >= MAX_SEARCH_MATCHES:
                hits.append(f"… 已达 {MAX_SEARCH_MATCHES} 条上限,请缩小范围")
                break
            try:
                text = file.read_bytes()[: self.max_read_bytes].decode("utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable; not what a text search is after
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{self._relative(file)}:{number}:{line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_MATCHES:
                        break

        return "\n".join(hits) if hits else f"没有匹配 '{pattern}' 的内容。"

    def _walk(self, root: Path):
        for entry in sorted(root.rglob("*")):
            if any(part in SKIP_DIRS for part in entry.parts):
                continue
            if entry.is_file():
                yield entry

    # ---- writing tools ------------------------------------------------

    def _guard_write(self) -> None:
        if not self.allow_write:
            raise ToolError("当前模式不允许写入(单次模式需要加 --allow-write)。")

    def _guard_protected(self, target: Path) -> None:
        if not self.protect:
            return
        reason = _protected_reason(target)
        if reason:
            raise ToolError(
                f"拒绝写入受保护的文件:{self._relative(target)}({reason})。"
                "确实需要时,把 config.json 里 tools.protect 设为 false。"
            )

    def _remember(
        self, path: Path, action: str, *, existed: bool, previous: str, written: str
    ) -> None:
        """Keep both versions; undo must not overwrite someone else's later edit."""
        self.writes.append(WriteRecord(path, action, existed, previous, written))
        if len(self.writes) > MAX_UNDO:
            del self.writes[:-MAX_UNDO]

    def undo_last(self) -> str | None:
        """Undo one write after confirming, without discarding failed attempts."""
        if not self.writes:
            return None
        self._guard_write()
        record = self.writes[-1]
        target = self._resolve(str(record.path), writable=True)
        self._guard_protected(target)
        if target != record.path or not target.is_file():
            raise ToolError("文件已移动或不存在,拒绝撤销以免覆盖别人的修改。")
        try:
            current = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolError(f"撤销前无法读取原文件:{exc}") from None
        if current != record.written:
            raise ToolError("文件在本次写入后又被修改,拒绝撤销以免覆盖新内容。")
        action = "撤销(删除新建的文件)" if not record.existed else "撤销"
        if not self.confirm(action, target, current, record.previous):
            raise ToolError("用户拒绝了这次撤销,文件未改动。")
        try:
            if record.existed:
                target.write_text(record.previous, encoding="utf-8")
            else:
                target.unlink()
        except OSError as exc:
            raise ToolError(f"撤销失败:{exc.strerror or exc}") from None
        self.writes.pop()
        if record.existed:
            return f"已把 {self._relative(target)} 还原到本次修改前"
        return f"已删除本次新建的 {self._relative(target)}"

    def _write_file(self, path: str, content: str) -> str:
        self._guard_write()
        target = self._resolve(path, writable=True)
        if target.is_dir():
            raise ToolError(f"{self._relative(target)} 是目录。")
        self._guard_protected(target)

        existed = target.exists()
        previous = ""
        if existed:
            try:
                previous = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ToolError(f"写入前无法读取原文件:{exc}") from None

        if not self.confirm("写入", target, previous, content):
            raise ToolError("用户拒绝了这次写入,文件未改动。")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._remember(target, "写入", existed=existed, previous=previous, written=content)
        return f"已写入 {self._relative(target)}({len(content)} 字符)。"

    def _edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        self._guard_write()
        target = self._resolve(path, writable=True)
        if not target.exists():
            raise ToolError(f"文件不存在:{self._relative(target)}")
        self._guard_protected(target)

        try:
            original = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{self._relative(target)} 不是 UTF-8 文本文件。") from None

        if not old_string:
            raise ToolError("old_string 不能为空。")

        occurrences = original.count(old_string)
        if occurrences == 0:
            raise ToolError("old_string 在文件中找不到,请先 read_file 确认原文。")
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_string 在文件中出现 {occurrences} 次,请补足上下文使其唯一,"
                "或设 replace_all=true。"
            )

        updated = original.replace(old_string, new_string) if replace_all else original.replace(
            old_string, new_string, 1
        )
        label = f"编辑({'全部替换' if replace_all else '单处替换'})"
        if not self.confirm(label, target, original, updated):
            raise ToolError("用户拒绝了这次编辑,文件未改动。")

        target.write_text(updated, encoding="utf-8")
        self._remember(target, label, existed=True, previous=original, written=updated)
        return f"已编辑 {self._relative(target)}(替换 {occurrences if replace_all else 1} 处)。"


def _matches_glob(name: str, pattern: str) -> bool:
    from fnmatch import fnmatch

    return fnmatch(name, pattern)


def auto_root(explicit: str | None) -> Path:
    """The tool workspace: an explicit --dir, else the directory the user
    launched from.

    Deliberately the CWD rather than the exe's folder. `cd D:\\proj` then
    running `X:\\agent\\agent.bat` should let the agent work on D:\\proj, not be
    trapped inside the USB stick.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd().resolve()
