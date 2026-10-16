"""Point the app at a temp directory so tests never touch the real config.json
or secrets.enc sitting next to the source."""

from pathlib import Path

import pytest

from selfagent import paths


@pytest.fixture
def app_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "app_dir", lambda: tmp_path)
    return tmp_path
