"""Offline tests for the transient-error backoff path in harness/backends.py.

No network: `GoogleGenAIBackend` is instantiated with a dummy key (the
`google.genai.Client` constructor doesn't make a network call), then its
internal `_client.models.generate_content` is monkeypatched to raise
`APIError`s (500/503/RESOURCE_EXHAUSTED) or return canned responses, so we can
prove the backoff/classification logic without hitting Google's API.
"""
import os
import sys
import time
import types as pytypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from harness.backends import (GoogleGenAIBackend, RateLimitError,
                              TransientBackendError, TRANSIENT_MAX_ATTEMPTS)
from harness.rlm import llm_query


def _make_backend(model: str = "gemma-4-26b-a4b-it") -> GoogleGenAIBackend:
    return GoogleGenAIBackend(api_key="dummy-key-not-used", model=model)


class _FakeResp:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None


def _api_error(code: int, status: str, message: str = "boom"):
    from google.genai import errors as genai_errors
    return genai_errors.APIError(code, {"status": status, "message": message})


def test_transient_500_then_success_returns_value_after_backoff(monkeypatch):
    backend = _make_backend()
    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def fake_generate_content(model, contents, config):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _api_error(500, "INTERNAL")
        return _FakeResp("hello after retries")

    backend._client.models.generate_content = fake_generate_content
    out = backend.generate("prompt", system="sys")

    assert out == "hello after retries"
    assert calls["n"] == 3
    assert len(sleeps) == 2, "should sleep once per retried attempt (2 retries before success)"
    # exponential-ish backoff: second sleep >= first (base*2**n growth, before jitter)
    assert sleeps[1] >= sleeps[0] - 0.3   # allow for jitter noise


def test_transient_exhausts_attempts_raises_typed_error(monkeypatch):
    backend = _make_backend()
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def always_503(model, contents, config):
        raise _api_error(503, "UNAVAILABLE")

    backend._client.models.generate_content = always_503
    with pytest.raises(TransientBackendError):
        backend.generate("prompt")
    # attempts == TRANSIENT_MAX_ATTEMPTS, sleeps == attempts - 1
    assert len(sleeps) == TRANSIENT_MAX_ATTEMPTS - 1


def test_empty_response_text_is_treated_as_transient(monkeypatch):
    backend = _make_backend()
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake(model, contents, config):
        calls["n"] += 1
        if calls["n"] < 2:
            return _FakeResp("")   # empty text -> transient, retry
        return _FakeResp("finally got text")

    backend._client.models.generate_content = fake
    out = backend.generate("prompt")
    assert out == "finally got text"
    assert calls["n"] == 2


def test_429_raises_ratelimiterror_immediately_no_backoff(monkeypatch):
    backend = _make_backend()
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fake_429(model, contents, config):
        calls["n"] += 1
        raise _api_error(429, "RESOURCE_EXHAUSTED")

    backend._client.models.generate_content = fake_429
    with pytest.raises(RateLimitError):
        backend.generate("prompt")
    assert calls["n"] == 1, "429 must not be retried with backoff"
    assert sleeps == []


def test_non_transient_api_error_propagates_unwrapped(monkeypatch):
    backend = _make_backend()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def fake_400(model, contents, config):
        raise _api_error(400, "INVALID_ARGUMENT")

    backend._client.models.generate_content = fake_400
    from google.genai import errors as genai_errors
    with pytest.raises(genai_errors.APIError):
        backend.generate("prompt")


def test_llm_query_returns_none_after_transient_exhausted(monkeypatch):
    """rlm.llm_query never raises for a TransientBackendError — it degrades to
    the sentinel path (returns None) and records the error on the log entry,
    exactly like any other backend hiccup."""
    backend = _make_backend()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def always_500(model, contents, config):
        raise _api_error(500, "INTERNAL")

    backend._client.models.generate_content = always_500

    from harness.logging_utils import HarnessLogger
    log = HarnessLogger()
    result = llm_query(backend, "give me json", leaf="test_leaf", logger=log)
    assert result is None
    assert len(log.entries) == 1
    assert log.entries[0].ok is False
    assert "TransientBackendError" in log.entries[0].error or \
        "INTERNAL" in log.entries[0].error


def test_llm_query_still_handles_ratelimiterror_as_before(monkeypatch):
    """llm_query's contract is unchanged: it never raises for a RateLimitError
    either (it's a RuntimeError caught by the same generic except as any other
    backend hiccup) — the leaf degrades to None/sentinel and the error is
    recorded on the log entry. RateLimitError propagating to the UI happens one
    layer up, from GoogleGenAIBackend.generate() directly when the app calls it
    outside of llm_query (see app.py's compile_jd try/except)."""
    backend = _make_backend()

    def fake_429(model, contents, config):
        raise _api_error(429, "RESOURCE_EXHAUSTED")

    backend._client.models.generate_content = fake_429

    from harness.logging_utils import HarnessLogger
    log = HarnessLogger()
    result = llm_query(backend, "give me json", leaf="test_leaf", logger=log)
    assert result is None
    assert log.entries[0].ok is False
    assert "RESOURCE_EXHAUSTED" in log.entries[0].error

    # Directly calling the backend (not through llm_query) still raises
    # RateLimitError immediately, with no backoff — this is the path compile_jd
    # would hit if it called backend.generate() itself.
    with pytest.raises(RateLimitError):
        backend.generate("prompt")


if __name__ == "__main__":
    import types
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            class _MP:
                def setattr(self, obj, name, value):
                    setattr(obj, name, value)
            try:
                fn(_MP())
            except TypeError:
                fn()
            print(f"PASS {name}")
    print("all ok")
