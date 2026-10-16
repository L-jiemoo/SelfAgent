"""Single-shot mode, for scripts and pipes.

Exit codes are part of the interface here: 0 success, 1 handled error, 130
interrupted. That makes `agent "..." || echo failed` behave the way a shell user
expects.
"""

from __future__ import annotations

from .engine import Engine
from .errors import SelfAgentError
from .render import Renderer


def run(engine: Engine, renderer: Renderer, prompt: str) -> int:
    try:
        engine.ask(prompt)
    except KeyboardInterrupt:
        renderer.finish()
        renderer.warning("已中断。")
        return 130
    except SelfAgentError as exc:
        renderer.finish()
        renderer.error(str(exc))
        return 1
    return 0
