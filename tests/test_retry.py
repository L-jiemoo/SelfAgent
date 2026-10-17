"""Retrying transient failures -- and refusing to retry anything else.

The distinction is the whole point: a rate limit or a dropped connection is
worth another go, while a bad key or a missing model will fail identically
every time and retrying only delays the message the user needs to read.
"""

from __future__ import annotations

import httpx
import pytest

from selfagent.providers import retry


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1"))


# ---- what counts as transient ----------------------------------------


def test_a_rate_limit_is_transient() -> None:
    import openai

    assert retry.is_transient(openai.RateLimitError("slow", response=_response(429), body=None))


def test_a_server_fault_is_transient() -> None:
    import openai

    assert retry.is_transient(
        openai.InternalServerError("boom", response=_response(500), body=None)
    )


def test_a_bad_key_is_not_transient() -> None:
    import openai

    assert not retry.is_transient(
        openai.AuthenticationError("nope", response=_response(401), body=None)
    )


def test_a_missing_model_is_not_transient() -> None:
    import openai

    assert not retry.is_transient(openai.NotFoundError("gone", response=_response(404), body=None))


def test_anthropic_failures_are_classified_too() -> None:
    import anthropic

    assert retry.is_transient(anthropic.RateLimitError("slow", response=_response(429), body=None))
    assert not retry.is_transient(
        anthropic.AuthenticationError("nope", response=_response(401), body=None)
    )


def test_an_unrelated_error_is_not_transient() -> None:
    assert not retry.is_transient(ValueError("nope"))


# ---- the loop --------------------------------------------------------


@pytest.fixture
def instant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry.time, "sleep", lambda _seconds: None)


def test_a_successful_open_is_not_repeated(monkeypatch: pytest.MonkeyPatch, instant: None) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: True)
    calls: list[int] = []

    def action() -> str:
        calls.append(1)
        return "ok"

    assert retry.open_with_retries(action, 3) == "ok"
    assert calls == [1]


def test_a_transient_failure_is_tried_again(monkeypatch: pytest.MonkeyPatch, instant: None) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: True)
    calls: list[int] = []

    def action() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert retry.open_with_retries(action, 3) == "ok"
    assert len(calls) == 3


def test_a_hard_failure_is_raised_at_once(monkeypatch: pytest.MonkeyPatch, instant: None) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: False)
    calls: list[int] = []

    def action() -> str:
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        retry.open_with_retries(action, 3)
    assert len(calls) == 1, "retrying a permanent failure only wastes the user's time"


def test_it_gives_up_after_the_budget(monkeypatch: pytest.MonkeyPatch, instant: None) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: True)
    calls: list[int] = []

    def action() -> str:
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry.open_with_retries(action, 2)
    assert len(calls) == 2


def test_the_callback_announces_each_upcoming_attempt(
    monkeypatch: pytest.MonkeyPatch, instant: None
) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: True)
    seen: list[tuple[int, int]] = []

    def action() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry.open_with_retries(action, 3, lambda a, t, d, e: seen.append((a, t)))
    assert seen == [(1, 3), (2, 3)]


def test_the_backoff_grows_and_then_stops() -> None:
    assert retry.delay_for(1) < retry.delay_for(2) < retry.delay_for(3)
    assert retry.delay_for(50) == retry.RETRY_MAX_DELAY


def test_one_attempt_means_no_retrying(monkeypatch: pytest.MonkeyPatch, instant: None) -> None:
    monkeypatch.setattr(retry, "is_transient", lambda exc: True)
    calls: list[int] = []

    def action() -> str:
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        retry.open_with_retries(action, 1)
    assert len(calls) == 1
