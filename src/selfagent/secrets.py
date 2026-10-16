"""Encrypted API-key store.

A USB stick is a thing that gets lost. Plaintext keys in a JSON file next to the
exe would mean anyone who picks it up inherits your API billing. So keys live in
secrets.enc, encrypted under a key derived from a passphrase typed once per
session.

Fernet is AES-128-CBC + HMAC-SHA256 with encrypt-then-MAC: decrypt() verifies the
MAC and raises InvalidToken rather than ever handing back tampered plaintext.
AES-128 is full strength here; the file's confidentiality rests on the KDF.

The derived Fernet handle is kept for the session so that saving an updated key
does not re-run the KDF, which is deliberately slow. It lives only in memory.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from . import paths
from .errors import SecretsError

KDF_NAME = "pbkdf2-hmac-sha256"
ITERATIONS = 600_000  # OWASP current guidance for PBKDF2-HMAC-SHA256
SALT_BYTES = 16
ENVELOPE_VERSION = 1


def derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def store_exists() -> bool:
    return paths.secrets_path().exists()


def _read_envelope(path=None) -> dict[str, Any]:
    path = path or paths.secrets_path()
    if not path.exists():
        raise SecretsError(
            f"未找到密钥库 {path.name}。运行 `agent --setup` 录入 API Key。"
        )
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SecretsError(
            f"{path.name} 已损坏(不是合法的 JSON)。"
            "若无法恢复,删除该文件后运行 `agent --setup` 重新录入。"
        ) from None
    if not isinstance(env, dict) or "salt" not in env or "token" not in env:
        raise SecretsError(f"{path.name} 内容不完整。删除后运行 `agent --setup` 重新录入。")
    return env


def _decode_envelope(env: dict[str, Any]) -> tuple[bytes, int, bytes]:
    try:
        salt = base64.b64decode(env["salt"], validate=True)
        token = env["token"].encode("ascii")
    except (ValueError, TypeError, AttributeError, UnicodeEncodeError):
        raise SecretsError("密钥库的字段无效,无法解密。") from None
    iterations = env.get("iterations", ITERATIONS)
    if not isinstance(iterations, int) or iterations <= 0:
        iterations = ITERATIONS
    return salt, iterations, token


def _write_envelope(env: dict[str, Any]) -> None:
    path = paths.secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


@dataclass
class SecretStore:
    """Decrypted keys for this session. Nothing here is ever written in clear."""

    keys: dict[str, str] = field(default_factory=dict)
    salt: bytes | None = None
    iterations: int = ITERATIONS
    _fernet: Fernet | None = None

    def api_key(self, provider: str) -> str | None:
        value = self.keys.get(provider)
        return value if value else None

    def has(self, provider: str) -> bool:
        return bool(self.keys.get(provider))

    def set(self, provider: str, api_key: str) -> None:
        self.keys[provider] = api_key.strip()

    def remove(self, provider: str) -> None:
        self.keys.pop(provider, None)

    @property
    def providers(self) -> list[str]:
        return sorted(self.keys)

    # ---- persistence --------------------------------------------------

    def _encrypt(self) -> bytes:
        assert self._fernet is not None
        return self._fernet.encrypt(
            json.dumps({"providers": self.keys}, ensure_ascii=False).encode("utf-8")
        )

    def save(self) -> None:
        """Re-encrypt in place. Requires an unlocked store."""
        if self._fernet is None or self.salt is None:
            raise SecretsError("密钥库未解锁,无法保存。")
        _write_envelope(
            {
                "version": ENVELOPE_VERSION,
                "kdf": KDF_NAME,
                "iterations": self.iterations,
                "salt": base64.b64encode(self.salt).decode("ascii"),
                "token": self._encrypt().decode("ascii"),
            }
        )


def create(passphrase: str) -> SecretStore:
    """Start a brand-new store bound to this passphrase."""
    salt = os.urandom(SALT_BYTES)
    store = SecretStore(
        keys={},
        salt=salt,
        iterations=ITERATIONS,
        _fernet=Fernet(derive_key(passphrase, salt, ITERATIONS)),
    )
    return store


def unlock(passphrase: str) -> SecretStore:
    """Decrypt the store. Raises SecretsError on a wrong passphrase."""
    env = _read_envelope()
    salt, iterations, token = _decode_envelope(env)
    fernet = Fernet(derive_key(passphrase, salt, iterations))

    try:
        plaintext = fernet.decrypt(token)
    except InvalidToken:
        raise SecretsError("口令错误,或密钥库已被篡改。") from None

    try:
        data = json.loads(plaintext.decode("utf-8"))
        providers = data.get("providers", {})
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        raise SecretsError("密钥库解密成功,但内容无法解析。") from None

    keys = {
        name: value
        for name, value in providers.items()
        if isinstance(name, str) and isinstance(value, str) and value
    }
    return SecretStore(keys=keys, salt=salt, iterations=iterations, _fernet=fernet)
