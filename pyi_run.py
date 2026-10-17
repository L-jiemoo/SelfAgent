"""Run PyInstaller with its cosmetic exe post-patches skipped.

Why this wrapper exists
-----------------------
PyInstaller links agent.exe and then, as its last step, rewrites the file
twice: ``set_exe_build_timestamp`` (stamps the PE header with the build time)
and ``update_exe_pe_checksum`` (recomputes the PE checksum). Both rewrite the
whole file with ``open(path, 'wb')``.

The Security Center on this machine denies that second write to the linked
exe. PyInstaller's retry loop gives up, the exception propagates, and the run
aborts *before* COLLECT -- so no bundle is produced at all, even though the
linked exe is already complete and runnable.

Both patches exist only so two rebuilds come out byte-identical; neither is
needed for the exe to run. Skipping them lets the build finish here. The
finished exe lands in ``andist/agent`` as a brand-new path that nothing
rewrites afterwards, so the block never comes into play again.

Usage: same arguments as ``python -m PyInstaller``. build.py calls this
instead of the module directly.
"""
from __future__ import annotations

try:
    from PyInstaller.utils.win32 import winutils
except ImportError:
    pass  # Not Windows: the post-processing step does not exist there.
else:
    winutils.set_exe_build_timestamp = lambda *args, **kwargs: None
    winutils.update_exe_pe_checksum = lambda *args, **kwargs: None

from PyInstaller.__main__ import run

if __name__ == "__main__":
    run()
