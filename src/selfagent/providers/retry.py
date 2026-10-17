"""Retrying transient provider failures.

Chinese networks drop connections to API endpoints often enough that "it just
failed once" is the normal case rather than the exception, and making the user
retype the question for it is a waste of their time.

The one rule that matters: only the *opening* of a request may be retried. A
stream that has already emitted tokens cannot be replayed without the user
seeing the same text twice, so mid-stream failures still surface as errors.
"""

from __future__ import annotations

import time
from typing import Any, Callable

RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 8.0
RETRY_ATTEMPTS = 2

OnRetry = Callable[[int, int, float, Exception], None]


def is_transient(exc: Exception) -> bool:
    """Rate limits, connection trouble and server-side faults -- nothing else.

    A bad key, a missing model or an endpoint that refuses `tools` will fail
    identically every time, so retrying only delays the error the user needs to
    read. Those are 4xx and are deliberately excluded.
    """
    try:
        import openai

        if isinstance(
            exc,
            (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError),
        ):
            return True
        if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
            return True
    except ImportError:  # pragma: no cover - hard dependency
        pass

    try:
        import anthropic

        if isinstance(
            exc,
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        ):
            return True
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            return True
    except ImportError:  # pragma: no cover - hard dependency
        pass

    return False


def delay_for(attempt: int) -> float:
    """Exponential backoff, capped. `attempt` is 1-based."""
    return min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** max(0, attempt - 1)))


def open_with_retries(
    action: Callable[[], Any],
    attempts: int = RETRY_ATTEMPTS,
    on_retry: OnRetry | None = None,
) -> Any:
    """Run `action()` -- which must only *open* a request -- retrying failures."""
    total = max(1, attempts)
    for attempt in range(1, total + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= total or not is_transient(exc):
                raise
            delay = delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, total, delay, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
