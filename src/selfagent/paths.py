"""Locate files relative to the application, never relative to the CWD.

The whole point of this module: the folder gets copied to a USB stick and the
drive letter changes from machine to machine. Anything resolved via os.getcwd()
or a hard-coded path breaks. sys._MEIPASS is also wrong here -- under onefile it
points at a temp extraction dir that is wiped between runs, so files written
there do not persist.
"""

from __future__ import annotations

import sys
from pathlib import Path

CONFIG_NAME = "config.json"
SECRETS_NAME = "secrets.enc"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Directory the user actually launched from.

    Frozen: the folder containing agent.exe. Dev: the repository root.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return app_dir() / CONFIG_NAME


def secrets_path() -> Path:
    return app_dir() / SECRETS_NAME


def resolve_data_dir(name: str) -> Path:
    """Resolve a possibly-relative data directory against the app dir.

    Absolute paths are honoured as-is, which is how a user redirects history
    onto a local disk instead of the (slow, possibly read-only) USB stick.
    """
    p = Path(name)
    return p if p.is_absolute() else app_dir() / p
