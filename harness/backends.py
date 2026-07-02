"""Model-agnostic backend layer (the RLM leaf caller).

A Backend is just: `generate(prompt, system=None) -> str`. The harness is written
against this protocol so the parity eval can swap models freely and tests can run
offline with a deterministic MockBackend.

Phase 1 ships GoogleGenAIBackend (Gemma 4 / Gemini via Google AI Studio, BYO key).
Gemma models have no native JSON-schema mode, so the harness — not the model —
guarantees structure (see rlm.py / coerce.py).
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

# Hosted model ids on Google AI Studio (verified: gemma-4 served via Gemini API).
GEMMA_MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it", "gemma-3-27b-it"]
GEMINI_MODELS = ["gemini-3-flash"]          # native response_schema path
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

# Per-call HTTP timeout (ms). Caps worst-case wall time so a rate-limited/slow
# call fails fast instead of the SDK backing off silently for minutes.
REQUEST_TIMEOUT_MS = 60000


class RateLimitError(RuntimeError):
    """Raised when the backend hits a 429 / RESOURCE_EXHAUSTED response.

    Lets the UI show the free-tier rate-cap guidance instead of a generic error.
    """


@runtime_checkable
class Backend(Protocol):
    name: str
    supports_response_schema: bool

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> str: ...


class MockBackend:
    """Deterministic offline backend for tests. `responder(prompt, system)->str`."""

    supports_response_schema = False

    def __init__(self, responder: Callable[[str, str | None], str], name: str = "mock"):
        self._responder = responder
        self.name = name
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        self.calls.append((prompt, system))
        return self._responder(prompt, system)


class GoogleGenAIBackend:
    """Gemma 4 / Gemini via the google-genai SDK (Google AI Studio key)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("GoogleGenAIBackend needs a Google AI Studio API key")
        from google import genai  # lazy: keep import cost out of tests
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.name = model
        self.model = model
        # Only Gemini models expose response_schema/system_instruction reliably.
        self.supports_response_schema = model.startswith("gemini")
        # Last-call token usage, surfaced for telemetry (None until a call runs).
        self.last_usage: dict | None = None

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        from google.genai import errors as genai_errors
        from google.genai import types
        cfg_kwargs = dict(
            temperature=temperature, max_output_tokens=max_tokens,
            # Per-request timeout so a rate-limited/slow call fails fast.
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        contents = prompt
        if system:
            if self.supports_response_schema:      # Gemini: real system slot
                cfg_kwargs["system_instruction"] = system
            else:                                   # Gemma: prepend to the prompt
                contents = f"{system}\n\n{prompt}"
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs))
        except genai_errors.APIError as exc:
            # Surface 429 / RESOURCE_EXHAUSTED as a typed error; let others propagate.
            code = getattr(exc, "code", None)
            status = str(getattr(exc, "status", "") or "")
            if code == 429 or "RESOURCE_EXHAUSTED" in status.upper() \
                    or "RESOURCE_EXHAUSTED" in str(exc).upper():
                raise RateLimitError(str(exc)) from exc
            raise
        # Capture token usage defensively (attribute names per google-genai contract).
        usage = getattr(resp, "usage_metadata", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return (resp.text or "").strip()

    def embed(self, texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
        """Hosted embeddings for the parity semantic-diff (same key)."""
        resp = self._client.models.embed_content(model=model, contents=texts)
        return [e.values for e in resp.embeddings]


def make_backend(api_key: str, model: str = DEFAULT_MODEL) -> Backend:
    return GoogleGenAIBackend(api_key=api_key, model=model)
