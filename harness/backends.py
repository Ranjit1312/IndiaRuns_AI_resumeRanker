"""Model-agnostic backend layer (the RLM leaf caller).

A Backend is just: `generate(prompt, system=None) -> str`. The harness is written
against this protocol so the parity eval can swap models freely and tests can run
offline with a deterministic MockBackend.

Phase 1 ships GoogleGenAIBackend (Gemma 4 / Gemini via Google AI Studio, BYO key).
Gemma models have no native JSON-schema mode, so the harness — not the model —
guarantees structure (see rlm.py / coerce.py).
"""
from __future__ import annotations

import random
import time
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


class TransientBackendError(RuntimeError):
    """Raised when the backend exhausts its retries on a transient server error.

    Transient = Google-side 500/503/INTERNAL/UNAVAILABLE/DEADLINE_EXCEEDED, or an
    empty response body. Distinct from RateLimitError: a 429 means "stop hitting
    us" (free-tier rate cap — backoff is pointless, raise immediately), while a
    transient 500 means "this one call misfired, try again shortly" — so
    generate() retries those internally with exponential backoff before giving up.
    """


# Backoff schedule for transient errors (NOT used for 429s — see RateLimitError).
TRANSIENT_MAX_ATTEMPTS = 3          # total tries, i.e. up to 2 retries after the first
TRANSIENT_BACKOFF_BASE_S = 0.8      # sleep = base * 2**n + jitter
TRANSIENT_BACKOFF_JITTER_S = 0.25
TRANSIENT_BACKOFF_CAP_S = 8.0       # per-sleep cap so worst case stays bounded

_TRANSIENT_STATUSES = {"INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
_TRANSIENT_CODES = {500, 503, 504}


def _is_transient(exc: "Exception", genai_errors) -> bool:
    """True if *exc* is a Google-side transient APIError (500/503/INTERNAL/
    UNAVAILABLE/DEADLINE_EXCEEDED) rather than a rate-limit or hard failure.

    Verified against google-genai 2.10.0: `errors.APIError` carries both `.code`
    (int, e.g. 500) and `.status` (str, e.g. "INTERNAL") — see
    google/genai/errors.py's APIError.__init__, which sets both from the
    response JSON. We check both since either can be populated depending on
    which layer raised (HTTP status vs. the API's own error envelope)."""
    if not isinstance(exc, genai_errors.APIError):
        return False
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    text = str(exc).upper()
    return (code in _TRANSIENT_CODES or status in _TRANSIENT_STATUSES
            or any(s in text for s in _TRANSIENT_STATUSES))


@runtime_checkable
class Backend(Protocol):
    name: str
    supports_response_schema: bool

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> str: ...


class MockBackend:
    """Deterministic offline backend for tests. `responder(prompt, system)->str`.

    Also supports `generate_multimodal` for OCR-path tests: pass
    `multimodal_responder(prompt, images, system) -> str` to exercise it, or
    leave it unset — calling `generate_multimodal` without one raises, same as
    any backend that doesn't implement it.
    """

    supports_response_schema = False

    def __init__(self, responder: Callable[[str, str | None], str], name: str = "mock",
                 multimodal_responder: Callable[[str, list, str | None], str] | None = None):
        self._responder = responder
        self._multimodal_responder = multimodal_responder
        self.name = name
        self.calls: list[tuple[str, str | None]] = []
        self.multimodal_calls: list[tuple[str, int, str | None]] = []
        self.last_usage: dict | None = None

    def generate(self, prompt: str, system: str | None = None,
                 temperature: float = 0.2, max_tokens: int = 2048) -> str:
        self.calls.append((prompt, system))
        return self._responder(prompt, system)

    def generate_multimodal(self, prompt: str, images: list[bytes], *,
                            system: str | None = None, mime_type: str = "image/png",
                            temperature: float = 0.2, max_tokens: int = 4096) -> str:
        self.multimodal_calls.append((prompt, len(images), system))
        if self._multimodal_responder is None:
            raise NotImplementedError("MockBackend: no multimodal_responder configured")
        return self._multimodal_responder(prompt, images, system)


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
        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = self._call_with_backoff(contents, config)
        return (resp.text or "").strip()

    def generate_multimodal(self, prompt: str, images: list[bytes], *,
                            system: str | None = None, mime_type: str = "image/png",
                            temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """Like `generate`, but attaches one or more images as `Part`s.

        Used for the Gemma-multimodal OCR path (harness/ingest.py's `_ocr_pdf`).
        Works for both gemma-* and gemini-* model ids at the SDK level: both
        accept a `contents` list mixing `types.Part.from_bytes(...)` image parts
        with a trailing text `Part` — the google-genai 2.10.0 API does not
        gate `Part.from_bytes` by model family. What is NOT verified without a
        live call is whether Google's *hosted* Gemma endpoint actually performs
        vision on the image bytes (Gemma 3 is documented multimodal; the gemma-4
        ids configured as GEMMA_MODELS in this repo were not live-tested here).
        If a given Gemma model id rejects/ignores image input at call time, this
        method still works unmodified for gemini-* ids (Gemini has verified
        multimodal support), so callers should wire an `ocr_fallback_backend`
        (see ingest.py) rather than relying on Gemma alone.
        """
        from google.genai import types
        cfg_kwargs = dict(
            temperature=temperature, max_output_tokens=max_tokens,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        text = prompt
        if system:
            if self.supports_response_schema:
                cfg_kwargs["system_instruction"] = system
            else:                                   # Gemma: prepend to the text part
                text = f"{system}\n\n{prompt}"
        parts = [types.Part.from_bytes(data=img, mime_type=mime_type) for img in images]
        parts.append(types.Part.from_text(text=text))
        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = self._call_with_backoff(parts, config)
        return (resp.text or "").strip()

    def _call_with_backoff(self, contents, config):
        """Shared call path for generate()/generate_multimodal(): dispatch to the
        SDK, classify errors, and retry transient ones with exponential backoff.

        - 429 / RESOURCE_EXHAUSTED -> RateLimitError, raised immediately (no
          backoff — a free-tier rate cap will not clear in a few seconds).
        - 500/503/INTERNAL/UNAVAILABLE/DEADLINE_EXCEEDED, or an empty `resp.text`
          -> transient: retried in-process up to TRANSIENT_MAX_ATTEMPTS times
          with base*2**n + jitter backoff, then raised as TransientBackendError.
        - anything else -> propagates as-is.
        """
        from google.genai import errors as genai_errors
        last_exc: Exception | None = None
        for attempt in range(TRANSIENT_MAX_ATTEMPTS):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except genai_errors.APIError as exc:
                code = getattr(exc, "code", None)
                status = str(getattr(exc, "status", "") or "")
                if code == 429 or "RESOURCE_EXHAUSTED" in status.upper() \
                        or "RESOURCE_EXHAUSTED" in str(exc).upper():
                    raise RateLimitError(str(exc)) from exc
                if _is_transient(exc, genai_errors):
                    last_exc = exc
                    if attempt < TRANSIENT_MAX_ATTEMPTS - 1:
                        self._sleep_backoff(attempt)
                        continue
                    raise TransientBackendError(str(exc)) from exc
                raise
            # Capture token usage defensively (attribute names per google-genai contract).
            usage = getattr(resp, "usage_metadata", None)
            self.last_usage = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
            }
            if not (resp.text or "").strip():
                # Empty output is also treated as transient (seen alongside 500s
                # on real runs) — retry the same way before giving up.
                last_exc = TransientBackendError("empty response text")
                if attempt < TRANSIENT_MAX_ATTEMPTS - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exc
            return resp
        # Unreachable in practice (loop always returns or raises), but keeps
        # mypy/pylint happy and fails loudly if the loop logic ever changes.
        raise last_exc or TransientBackendError("generate_content failed with no response")

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        sleep_s = min(TRANSIENT_BACKOFF_CAP_S,
                      TRANSIENT_BACKOFF_BASE_S * (2 ** attempt)
                      + random.uniform(0, TRANSIENT_BACKOFF_JITTER_S))
        time.sleep(sleep_s)

    def embed(self, texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
        """Hosted embeddings for the parity semantic-diff (same key)."""
        resp = self._client.models.embed_content(model=model, contents=texts)
        return [e.values for e in resp.embeddings]


def make_backend(api_key: str, model: str = DEFAULT_MODEL) -> Backend:
    return GoogleGenAIBackend(api_key=api_key, model=model)
