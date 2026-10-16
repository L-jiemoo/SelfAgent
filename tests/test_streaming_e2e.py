"""End-to-end through a fake OpenAI-compatible endpoint.

This is the test that would catch a broken adapter, a renderer that swallows
tokens, or a session that forgets to persist -- none of which unit tests on
config/registry can see. It stands up a real HTTP server speaking SSE so the
openai SDK is exercised for real, including its retry-on-400 path.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from selfagent import secrets
from selfagent.config import load_config
from selfagent.engine import Engine
from selfagent.render import Renderer


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)

        if self.server.reject_stream_options and "stream_options" in body:
            payload = json.dumps(
                {"error": {"message": "unknown field stream_options"}}
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        for piece in self.server.pieces:
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))

        final = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:  # silence the server
        pass


class FakeServer:
    def __init__(self, pieces, reject_stream_options=False):
        self.pieces = pieces
        self.reject_stream_options = reject_stream_options
        self.requests: list[dict] = []
        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.requests = self.requests
        self._httpd.pieces = pieces
        self._httpd.reject_stream_options = reject_stream_options
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def write_config(app_dir: Path, base_url: str, *, history_dir: str = "history") -> None:
    config = {
        "version": 1,
        "default_provider": "fake",
        "default_model": "fake-model",
        "providers": {"fake": {"kind": "openai", "base_url": base_url, "models": ["fake-model"]}},
        "aliases": {},
        "generation": {"max_tokens": 128, "temperature": None, "system": "be brief"},
        "ui": {"stream": True, "color": False, "show_thinking": False},
        "history": {"enabled": True, "dir": history_dir, "max_turns": 10},
    }
    (app_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )


def store_with_key(app_dir: Path) -> secrets.SecretStore:
    store = secrets.create("pw")
    store.set("fake", "sk-test")
    store.save()
    return secrets.unlock("pw")


def make_engine(app_dir: Path, out: io.StringIO, **kwargs) -> Engine:
    config = load_config()
    renderer = Renderer(color=False, out=out, err=out)
    return Engine(config, store_with_key(app_dir), renderer, **kwargs)


def test_streaming_produces_the_full_reply(app_dir: Path) -> None:
    with FakeServer(["你", "好", ",世界"]) as server:
        write_config(app_dir, server.base_url)
        out = io.StringIO()
        engine = make_engine(app_dir, out)

        reply = engine.ask("hi")

    assert reply == "你好,世界"
    # finish() closes the line, so redirected output ends with a newline.
    assert out.getvalue() == "你好,世界\n"


def test_request_carries_system_prompt_and_no_temperature(app_dir: Path) -> None:
    with FakeServer(["ok"]) as server:
        write_config(app_dir, server.base_url)
        make_engine(app_dir, io.StringIO()).ask("hi")

    sent = server.requests[0]
    assert sent["model"] == "fake-model"
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    # temperature: null in config must mean the field is absent, not zero --
    # thinking models reject sampling params outright.
    assert "temperature" not in sent


def test_falls_back_when_endpoint_rejects_stream_options(app_dir: Path) -> None:
    with FakeServer(["recovered"], reject_stream_options=True) as server:
        write_config(app_dir, server.base_url)
        out = io.StringIO()
        reply = make_engine(app_dir, out).ask("hi")

    assert reply == "recovered"
    assert len(server.requests) == 2, "should have retried exactly once"
    assert "stream_options" in server.requests[0]
    assert "stream_options" not in server.requests[1]


def test_usage_is_accumulated(app_dir: Path) -> None:
    with FakeServer(["a", "b"]) as server:
        write_config(app_dir, server.base_url)
        engine = make_engine(app_dir, io.StringIO())
        engine.ask("hi")

    assert engine.session.usage.input_tokens == 7
    assert engine.session.usage.output_tokens == 3


def test_multi_turn_history_is_sent_back(app_dir: Path) -> None:
    with FakeServer(["first"]) as server:
        write_config(app_dir, server.base_url)
        engine = make_engine(app_dir, io.StringIO())
        engine.ask("one")
        engine.ask("two")

    second = server.requests[1]["messages"]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "assistant", "user"]
    assert second[-1]["content"] == "two"


def test_history_file_is_written(app_dir: Path) -> None:
    with FakeServer(["hello"]) as server:
        write_config(app_dir, server.base_url)
        make_engine(app_dir, io.StringIO()).ask("hi")

    files = list((app_dir / "history").glob("*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert [entry["type"] for entry in lines] == [
        "session_start",
        "user",
        "assistant",
    ]
    assert lines[2]["text"] == "hello"


def test_history_failure_degrades_without_breaking_the_turn(app_dir: Path) -> None:
    # A file where the history directory should be: mkdir will fail with
    # NotADirectoryError, which must not take the conversation down with it.
    (app_dir / "blocked").write_text("not a directory", encoding="utf-8")

    with FakeServer(["still works"]) as server:
        write_config(app_dir, server.base_url, history_dir="blocked")
        warnings: list[str] = []
        config = load_config()
        renderer = Renderer(color=False, out=io.StringIO(), err=io.StringIO())
        engine = Engine(
            config,
            store_with_key(app_dir),
            renderer,
            persist=True,
        )

        reply = engine.ask("hi")

    assert reply == "still works"


def test_no_history_flag_skips_writing(app_dir: Path) -> None:
    with FakeServer(["x"]) as server:
        write_config(app_dir, server.base_url)
        make_engine(app_dir, io.StringIO(), persist=False).ask("hi")

    assert not (app_dir / "history").exists()


def test_reply_reaches_a_pipe_with_no_escape_codes(app_dir: Path) -> None:
    with FakeServer(["plain text"]) as server:
        write_config(app_dir, server.base_url)
        out = io.StringIO()
        make_engine(app_dir, out).ask("hi")

    assert "\x1b" not in out.getvalue()
