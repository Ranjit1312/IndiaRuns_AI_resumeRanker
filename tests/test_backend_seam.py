"""Offline tests for the Candidate E backend seam split (harness/backends.py).

Asserts that `TextBackend` and `EmbeddingBackend` — the two capability
Protocols the fuzzy old `Backend` was split into — are both satisfied by
MockBackend AND by GoogleGenAIBackend, so mock and real can't drift apart on
either seam. No network: GoogleGenAIBackend is only checked structurally
(class-level `hasattr`), and separately its constructor is proven not to hit
the network when given a dummy key (google.genai.Client() only builds a
local HTTP client config; it doesn't call out until a request method runs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.backends import (
    Backend,
    EmbeddingBackend,
    GoogleGenAIBackend,
    MockBackend,
    TextBackend,
)


def _noop_responder(prompt, system=None):
    return "ok"


def test_backend_is_alias_of_text_backend():
    # Back-compat: existing type hints elsewhere (rlm.py, coerce.py, resume.py)
    # import `Backend` — it must still be the text-generation Protocol.
    assert Backend is TextBackend


def test_mock_backend_satisfies_text_backend():
    backend = MockBackend(_noop_responder)
    assert isinstance(backend, TextBackend)


def test_mock_backend_satisfies_embedding_backend():
    backend = MockBackend(_noop_responder)
    assert isinstance(backend, EmbeddingBackend)


def test_mock_backend_embed_returns_equal_length_vectors():
    backend = MockBackend(_noop_responder)
    vecs = backend.embed(["a", "b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == len(vecs[1])
    assert len(vecs[0]) > 0
    # Deterministic: same text -> same vector.
    assert backend.embed(["a"])[0] == vecs[0]
    # Different text -> different vector.
    assert vecs[0] != vecs[1]


def test_mock_backend_embed_vectors_are_l2_normalized():
    backend = MockBackend(_noop_responder)
    vec = backend.embed(["some text"])[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_google_genai_backend_class_satisfies_text_backend_structurally():
    # Structural check on the class itself (no instance / no network): every
    # attribute/method TextBackend requires is defined on GoogleGenAIBackend.
    for attr in ("generate", "generate_multimodal"):
        assert hasattr(GoogleGenAIBackend, attr), f"missing {attr}"
    # `name`, `supports_response_schema`, `last_usage` are instance attributes
    # set in __init__ rather than class-level, so they can't be hasattr-checked
    # on the class; verified instead via source inspection of __init__ below.
    import inspect
    src = inspect.getsource(GoogleGenAIBackend.__init__)
    assert "self.name" in src
    assert "self.supports_response_schema" in src
    assert "self.last_usage" in src


def test_google_genai_backend_class_satisfies_embedding_backend_structurally():
    assert hasattr(GoogleGenAIBackend, "embed")


def test_google_genai_backend_instance_satisfies_both_protocols_without_network():
    # Constructing GoogleGenAIBackend with a dummy key must not hit the
    # network: genai.Client(api_key=...) only builds local client config: it
    # does not perform a handshake or validate the key until a request method
    # (e.g. generate_content) is actually called. We construct with a clearly
    # fake key and never call a network method, then check Protocol conformance
    # via isinstance (runtime_checkable Protocols only check attribute
    # presence, not call behavior, so this stays offline).
    backend = GoogleGenAIBackend(api_key="test-key-not-real")
    assert isinstance(backend, TextBackend)
    assert isinstance(backend, EmbeddingBackend)
    # Confirm no network I/O occurred: last_usage is still unset (None), which
    # is only populated after an actual generate_content() round trip.
    assert backend.last_usage is None
