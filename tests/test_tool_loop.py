"""The full agent loop against a scripted OpenAI-compatible endpoint.

This is where the risky code lives: assembling tool calls out of streamed
fragments, feeding results back in the shape each API demands, and stopping.
The fake endpoint is scripted per-request so a test can describe exactly the
delta shapes it wants to prove the adapter survives -- including the malformed
ones real providers emit.
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
from selfagent.tools import Toolbox


def d(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def text(piece: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": piece}}]}


def tool_delta(index, *, id=None, name=None, arguments=None) -> dict:
    call: dict = {"function": {}}
    if index is not None:
        call["index"] = index
    if id is not None:
        call["id"] = id
    if name is not None:
        call["function"]["name"] = name
    if arguments is not None:
        call["function"]["arguments"] = arguments
    return {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}


def tool_deltas(calls) -> dict:
    """Several tool_call fragments in ONE delta, as GLM does."""
    entries = []
    for call in calls:
        entry: dict = {"function": {}}
        if call.get("index") is not None:
            entry["index"] = call["index"]
        if call.get("id"):
            entry["id"] = call["id"]
        if call.get("name"):
            entry["function"]["name"] = call["name"]
        if call.get("arguments"):
            entry["function"]["arguments"] = call["arguments"]
        entries.append(entry)
    return {"choices": [{"index": 0, "delta": {"tool_calls": entries}}]}


def finish(reason: str) -> dict:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def usage() -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)

        script = self.server.scripts.pop(0) if self.server.scripts else [finish("stop")]

        if isinstance(script, dict) and script.get("status") != 200:
            payload = json.dumps(
                {"error": {"message": script.get("message", "bad request")}}
            ).encode()
            self.send_response(script["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in script:
            self.wfile.write(d(chunk))
        self.wfile.write(d(usage()))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:
        pass


class Server:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests: list[dict] = []
        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.requests = self.requests
        self._httpd.scripts = self.scripts
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


class Approve:
    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.calls: list[tuple] = []

    def __call__(self, action, path, old_text, new_text) -> bool:
        self.calls.append((action, path, old_text, new_text))
        return self.approve


def setup_env(app_dir: Path, server: Server, workspace: Path, *, max_steps: int = 10,
              allow_write: bool = True, approve: bool = True):
    config = {
        "version": 1,
        "default_provider": "fake",
        "default_model": "fake-model",
        "providers": {
            "fake": {
                "kind": "openai",
                "base_url": server.base_url,
                "models": ["fake-model"],
                "tools": True,
            }
        },
        "aliases": {},
        "generation": {"max_tokens": 256, "temperature": None, "system": "be brief"},
        "ui": {"stream": True, "color": False, "show_thinking": False},
        "history": {"enabled": False, "dir": "history", "max_turns": 10},
        "tools": {
            "enabled": True,
            "workspace": str(workspace),
            "confirm_writes": True,
            "max_steps": max_steps,
            "max_read_bytes": 65536,
        },
    }
    (app_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    store = secrets.create("pw")
    store.set("fake", "sk-test")
    store.save()

    confirm = Approve(approve)
    toolbox = Toolbox(root=workspace, confirm=confirm, allow_write=allow_write)
    renderer = Renderer(color=False, out=io.StringIO(), err=io.StringIO())
    engine = Engine(load_config(), secrets.unlock("pw"), renderer, toolbox=toolbox)
    return engine, confirm, renderer


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "data.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    return root


# ---- the loop --------------------------------------------------------


def test_model_reads_a_file_then_answers(app_dir: Path, workspace: Path) -> None:
    with Server(
        [
            # Turn 1: the model asks for a file.
            [
                tool_delta(0, id="c1", name="read_file"),
                tool_delta(0, arguments='{"path"'),
                tool_delta(0, arguments=': "data.txt"}'),
                finish("tool_calls"),
            ],
            # Turn 2: it has the contents and answers.
            [text("The file says alpha and beta."), finish("stop")],
        ]
    ) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        reply = engine.ask("what is in data.txt?")

    assert reply == "The file says alpha and beta."
    assert len(server.requests) == 2

    # The assistant turn must echo its calls, and the result must come back as
    # its own tool-role message keyed by the same id.
    second = server.requests[1]["messages"]
    assistant = next(m for m in second if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"

    tool_message = next(m for m in second if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "c1"
    assert "alpha" in tool_message["content"]


def test_fragmented_arguments_are_reassembled(app_dir: Path, workspace: Path) -> None:
    with Server(
        [
            [
                tool_delta(0, id="c1", name="read_file"),
                tool_delta(0, arguments='{"pa'),
                tool_delta(0, arguments='th": "da'),
                tool_delta(0, arguments='ta.txt"}'),
                finish("tool_calls"),
            ],
            [text("ok"), finish("stop")],
        ]
    ) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        engine.ask("read it")

    tool_message = next(m for m in server.requests[1]["messages"] if m["role"] == "tool")
    assert "alpha" in tool_message["content"], "arguments never reassembled"


def test_parallel_calls_without_index_are_kept_separate(app_dir: Path, workspace: Path) -> None:
    # Zhipu GLM emits several calls in one delta with no `index`. Folding them
    # into one slot concatenates their arguments into invalid JSON.
    with Server(
        [
            [
                tool_deltas(
                    [
                        {"id": "c1", "name": "read_file", "arguments": '{"path":"data.txt"}'},
                        {"id": "c2", "name": "list_dir", "arguments": '{"path":"."}'},
                    ]
                ),
                finish("tool_calls"),
            ],
            [text("done"), finish("stop")],
        ]
    ) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        engine.ask("both")

    assistant = next(
        m for m in server.requests[1]["messages"] if m["role"] == "assistant"
    )
    names = [c["function"]["name"] for c in assistant["tool_calls"]]
    assert names == ["read_file", "list_dir"]

    results = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert len(results) == 2
    assert "alpha" in results[0]["content"]


def test_every_call_is_answered_when_one_is_refused(app_dir: Path, workspace: Path) -> None:
    # Anthropic 400s if any tool_use goes unanswered, so a refusal has to come
    # back as an ordinary (error-flagged) result rather than being dropped.
    with Server(
        [
            [
                tool_delta(0, id="w1", name="write_file"),
                tool_delta(0, arguments='{"path":"new.txt","content":"hi"}'),
                finish("tool_calls"),
            ],
            [text("understood"), finish("stop")],
        ]
    ) as server:
        engine, _, _ = setup_env(app_dir, server, workspace, approve=False)
        engine.ask("make a file")

    results = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "w1"
    assert "拒绝" in results[0]["content"]
    assert not (workspace / "new.txt").exists()


def test_approved_write_actually_lands(app_dir: Path, workspace: Path) -> None:
    with Server(
        [
            [
                tool_delta(0, id="w1", name="write_file"),
                tool_delta(0, arguments='{"path":"out.txt","content":"written"}'),
                finish("tool_calls"),
            ],
            [text("done"), finish("stop")],
        ]
    ) as server:
        engine, confirm, _ = setup_env(app_dir, server, workspace)
        engine.ask("write it")

    assert (workspace / "out.txt").read_text(encoding="utf-8") == "written"
    assert len(confirm.calls) == 1


def test_the_loop_stops_at_max_steps(app_dir: Path, workspace: Path) -> None:
    # A model that re-calls forever must not bill forever.
    endless = [
        [
            tool_delta(0, id=f"c{i}", name="read_file"),
            tool_delta(0, arguments='{"path":"data.txt"}'),
            finish("tool_calls"),
        ]
        for i in range(20)
    ]
    with Server(endless) as server:
        engine, _, renderer = setup_env(app_dir, server, workspace, max_steps=3)
        engine.ask("loop forever")

    assert len(server.requests) == 3
    assert "上限" in renderer.err.getvalue()


# ---- degradation -----------------------------------------------------


def test_endpoint_rejecting_tools_is_retried_as_plain_chat(
    app_dir: Path, workspace: Path
) -> None:
    # Sending `tools` can break an endpoint that would otherwise work, and there
    # is no standard error code for it.
    with Server(
        [
            {"status": 400, "message": "unsupported parameter: tools"},
            [text("plain answer"), finish("stop")],
        ]
    ) as server:
        engine, _, renderer = setup_env(app_dir, server, workspace)
        reply = engine.ask("hello")

    assert reply == "plain answer"
    assert "tools" in server.requests[0]
    assert "tools" not in server.requests[1]
    assert "纯文本" in renderer.err.getvalue()


def test_degradation_sticks_for_the_session(app_dir: Path, workspace: Path) -> None:
    with Server(
        [
            {"status": 400, "message": "tools not supported"},
            [text("one"), finish("stop")],
            [text("two"), finish("stop")],
        ]
    ) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        engine.ask("first")
        engine.ask("second")

    assert all("tools" not in request for request in server.requests[1:])


def test_provider_capability_flag_disables_tools(app_dir: Path, workspace: Path) -> None:
    with Server([[text("no tools here"), finish("stop")]]) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        engine.target.provider.tools = False
        reply = engine.ask("hello")

    assert reply == "no tools here"
    assert "tools" not in server.requests[0]


def test_tools_are_not_sent_when_disabled(app_dir: Path, workspace: Path) -> None:
    with Server([[text("hi"), finish("stop")]]) as server:
        engine, _, _ = setup_env(app_dir, server, workspace)
        engine.tools_enabled = False
        engine.ask("hello")

    assert "tools" not in server.requests[0]


def test_tool_activity_goes_to_stderr_not_stdout(app_dir: Path, workspace: Path) -> None:
    with Server(
        [
            [
                tool_delta(0, id="c1", name="read_file"),
                tool_delta(0, arguments='{"path":"data.txt"}'),
                finish("tool_calls"),
            ],
            [text("the answer"), finish("stop")],
        ]
    ) as server:
        engine, _, renderer = setup_env(app_dir, server, workspace)
        engine.ask("read it")

    assert "read_file" in renderer.err.getvalue()
    assert "read_file" not in renderer.out.getvalue()
