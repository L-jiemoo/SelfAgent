import base64
import json
from pathlib import Path

import pytest

from selfagent import secrets
from selfagent.errors import SecretsError


def fresh_store(app_dir: Path) -> secrets.SecretStore:
    store = secrets.create("correct horse battery staple")
    store.save()
    return store


def test_roundtrip(app_dir: Path) -> None:
    store = fresh_store(app_dir)
    store.set("deepseek", "sk-abc123")
    store.set("anthropic", "sk-ant-xyz")
    store.save()

    reopened = secrets.unlock("correct horse battery staple")
    assert reopened.api_key("deepseek") == "sk-abc123"
    assert reopened.api_key("anthropic") == "sk-ant-xyz"


def test_keys_are_not_written_in_plaintext(app_dir: Path) -> None:
    store = fresh_store(app_dir)
    store.set("deepseek", "sk-SUPER-SECRET-VALUE")
    store.save()

    raw = (app_dir / "secrets.enc").read_text(encoding="utf-8")
    assert "sk-SUPER-SECRET-VALUE" not in raw


def test_wrong_passphrase_is_rejected(app_dir: Path) -> None:
    fresh_store(app_dir)
    with pytest.raises(SecretsError, match="口令错误"):
        secrets.unlock("wrong")


def test_tampered_ciphertext_is_rejected(app_dir: Path) -> None:
    store = fresh_store(app_dir)
    store.set("deepseek", "sk-abc")
    store.save()

    path = app_dir / "secrets.enc"
    env = json.loads(path.read_text(encoding="utf-8"))
    # Fernet tokens are URL-safe base64, not standard base64.
    raw = bytearray(base64.urlsafe_b64decode(env["token"]))
    raw[-1] ^= 0x01  # flip a bit in the HMAC tag
    env["token"] = base64.urlsafe_b64encode(bytes(raw)).decode("ascii")
    path.write_text(json.dumps(env), encoding="utf-8")

    with pytest.raises(SecretsError):
        secrets.unlock("correct horse battery staple")


def test_corrupt_file_reports_cleanly(app_dir: Path) -> None:
    (app_dir / "secrets.enc").write_text("not json at all", encoding="utf-8")
    with pytest.raises(SecretsError, match="损坏"):
        secrets.unlock("whatever")


def test_missing_file_reports_setup_hint(app_dir: Path) -> None:
    assert not secrets.store_exists()
    with pytest.raises(SecretsError, match="--setup"):
        secrets.unlock("whatever")


def test_salt_changes_per_store(app_dir: Path) -> None:
    secrets.create("pw").save()
    first = json.loads((app_dir / "secrets.enc").read_text(encoding="utf-8"))
    secrets.create("pw").save()
    second = json.loads((app_dir / "secrets.enc").read_text(encoding="utf-8"))
    assert first["salt"] != second["salt"]


def test_remove(app_dir: Path) -> None:
    store = fresh_store(app_dir)
    store.set("kimi", "sk-1")
    store.remove("kimi")
    assert not store.has("kimi")


def test_envelope_records_kdf_params(app_dir: Path) -> None:
    fresh_store(app_dir)
    env = json.loads((app_dir / "secrets.enc").read_text(encoding="utf-8"))
    assert env["kdf"] == "pbkdf2-hmac-sha256"
    assert env["iterations"] >= 600_000
