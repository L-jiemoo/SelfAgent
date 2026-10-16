"""Build agent.exe into andist\\agent\\ as a self-contained onedir bundle.

Run from the project root:  python build.py

Why onedir and not onefile: onefile re-extracts the whole bundle into %TEMP% on
every launch, which costs seconds off a USB stick and -- more importantly -- is
blocked outright on machines whose policy forbids executing from %TEMP%. onedir
runs in place and starts in well under a second. Both are equally "no install
needed" on the target machine, which is the actual requirement.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DIST = ROOT / "andist"
TARGET = DIST / "agent"
WORK = ROOT / "build"

# Copied next to the exe so the folder is usable the moment it is copied.
ALWAYS_ASSETS = ("agent.bat",)

# Readme may legitimately be .md or .txt depending on how the user renamed it.
README_NAMES = ("README.md", "README.txt")

# Things the user accumulates in the shipped folder. A rebuild wipes andist, so
# these are moved aside first and put back after -- losing a secrets.enc means
# re-entering every API key, and losing skills means re-installing them.
PRESERVE = ("secrets.enc", "config.json", "history", "skills")

# collect_all pulls submodules, data files and binaries for packages that load
# things dynamically. copy_metadata is separate and easy to forget: openai and
# anthropic read their own version through importlib.metadata at import time, and
# collect_all does NOT copy that.
COLLECT_ALL = ("anthropic", "openai", "cryptography")
COLLECT_SUBMODULES = ("pydantic", "httpx")
COPY_METADATA = ("anthropic", "openai", "pydantic", "rich")
HIDDEN_IMPORTS = ("httpx", "certifi")

# None of these are used; excluding them keeps tens of megabytes out of the
# bundle if anything reaches for them indirectly.
EXCLUDES = ("tkinter", "numpy", "matplotlib", "pytest", "IPython", "pandas")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=ROOT)


def ensure_venv() -> Path:
    python = venv_python()
    if not python.exists():
        print("创建虚拟环境 .venv ...", flush=True)
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(VENV)
    return python


def install_deps(python: Path) -> None:
    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            "requirements.txt",
            "-r",
            "requirements-dev.txt",
        ]
    )


def find_readme() -> Path | None:
    for name in README_NAMES:
        candidate = ROOT / name
        if candidate.exists():
            return candidate
    return None


# ---- user data that must survive a rebuild ---------------------------


def _config_is_user_modified(shipped: Path) -> bool:
    """True if the shipped config.json differs in *content* from the repo's.

    Compared parsed rather than byte-for-byte: the shipped file is a byte copy
    of the repo's, so any real difference is a real edit -- but a user who opens
    it in Notepad and saves without changing anything would otherwise look
    modified forever and stop receiving config updates.
    """
    repo = ROOT / "config.json"
    if not repo.exists():
        return True
    try:
        shipped_data = json.loads(shipped.read_text(encoding="utf-8"))
        repo_data = json.loads(repo.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    return shipped_data != repo_data


def preserve_user_data() -> Path | None:
    """Move accumulated files out of the build target. Returns the staging dir.

    Must run before build(), which deletes andist wholesale.
    """
    if not TARGET.exists():
        return None

    staging = Path(tempfile.mkdtemp(prefix="selfagent-preserve-"))
    kept: list[str] = []
    for name in PRESERVE:
        source = TARGET / name
        if not source.exists():
            continue
        if name == "config.json" and not _config_is_user_modified(source):
            continue
        shutil.move(str(source), str(staging / name))
        kept.append(name)

    if not kept:
        shutil.rmtree(staging, ignore_errors=True)
        return None

    print(f"保留已有数据(构建后会还原): {', '.join(kept)}", flush=True)
    return staging


def restore_user_data(staging: Path | None) -> None:
    if staging is None or not staging.exists():
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    for item in staging.iterdir():
        destination = TARGET / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination, ignore_errors=True)
            else:
                destination.unlink()
        shutil.move(str(item), str(destination))
    shutil.rmtree(staging, ignore_errors=True)


# ---- build -----------------------------------------------------------


def build(python: Path) -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(DIST, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onedir", "--console",
        "--name", "agent",
        "--distpath", "andist",
        "--workpath", "build",
        # Without this PyInstaller drops the generated .spec into the repo root.
        "--specpath", "build",
        "--paths", "src",
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for pkg in COLLECT_SUBMODULES:
        cmd += ["--collect-submodules", pkg]
    for pkg in COPY_METADATA:
        cmd += ["--copy-metadata", pkg]
    for module in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", module]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]
    cmd.append("main.py")

    run(cmd)


def copy_assets(has_user_config: bool) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)

    for name in ALWAYS_ASSETS:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, TARGET / name)
        else:
            print(f"提示: 未找到 {name},跳过。", flush=True)

    # Starter skills, so a fresh USB stick demonstrates the format. A user's own
    # skills are restored on top of these afterwards and therefore win.
    starters = ROOT / "skills"
    if starters.is_dir():
        shutil.copytree(starters, TARGET / "skills", dirs_exist_ok=True)
        print("已放入示例技能 skills\\。不需要可直接删掉。", flush=True)

    readme = find_readme()
    if readme is None:
        print("提示: 未找到 README.md 或 README.txt,跳过。", flush=True)
    else:
        shutil.copy2(readme, TARGET / readme.name)

    if has_user_config:
        print("保留你修改过的 config.json,未用仓库版本覆盖。", flush=True)
    else:
        shutil.copy2(ROOT / "config.json", TARGET / "config.json")


def report() -> None:
    exe = TARGET / "agent.exe"
    if not exe.exists():
        print("构建似乎失败了:未找到 andist\\agent\\agent.exe", file=sys.stderr)
        return

    total = sum(f.stat().st_size for f in TARGET.rglob("*") if f.is_file())
    print()
    print(f"构建完成: {exe}")
    print(f"整个文件夹: {total / 1024 / 1024:.1f} MB")
    print()
    print("把 andist\\agent 这一个文件夹整个拷到U盘即可。")
    print("若U盘上已有 secrets.enc / history,复制时保留它们,不要覆盖。")
    print('目标电脑上打开 cmd 运行:X:\\agent\\agent.bat "你好"')


def main() -> int:
    if not (ROOT / "main.py").exists():
        print("请在项目根目录运行本脚本。", file=sys.stderr)
        return 1

    python = ensure_venv()
    install_deps(python)

    preserved = preserve_user_data()
    try:
        build(python)
        copy_assets(has_user_config=(preserved is not None and (preserved / "config.json").exists()))
    finally:
        # Even a failed build must not cost the user their keys.
        restore_user_data(preserved)

    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
