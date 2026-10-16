"""Frozen entry point.

Kept trivial on purpose: PyInstaller needs a top-level script, and when frozen
sys.executable points at the built exe, which is what paths.py uses to find
config.json next to it.
"""

from selfagent.cli import main

raise SystemExit(main())
