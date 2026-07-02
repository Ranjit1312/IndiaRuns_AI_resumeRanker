"""Depth-1 RLM engine: the JD as an external variable + focused leaf calls.

Faithful to arXiv 2512.24601 at the scale that matters for a short JD: the input
lives as a variable the *root* (our deterministic code) inspects and slices, and
work is delegated to leaf `llm_query` calls that each see only a relevant snippet
+ one sub-schema. The paper's lesson — small models win by learning *when/how to
delegate*, not by leaf reasoning — is why the root is code, not the model.
"""
from __future__ import annotations

import re
from typing import Any

from .backends import Backend
from .jsonutil import extract_json
from .prompts import SYSTEM

MAX_SNIPPET = 6000   # cap each leaf's view (RLM "bounded output" spirit)

# canonical section -> header keywords that introduce it
_SECTION_KEYS = {
    "responsibilities": ["responsibilit", "what you'll do", "what you will do",
                         "role", "the job", "key job", "day in the life", "about the role"],
    "qualifications": ["qualification", "requirement", "what you'll bring",
                       "what you will bring", "who you are", "skills", "we're looking for",
                       "we are looking for", "basic qualification", "minimum qualification",
                       "preferred qualification", "you have", "you'll need"],
    "location": ["location", "where you", "remote", "hybrid", "onsite", "office"],
    "about": ["about the team", "about us", "who we are", "company"],
    "comp": ["compensation", "salary", "benefits", "pay range"],
}

_HEADER_RE = re.compile(r"^\s{0,4}([A-Z][^\n]{0,60})\s*:?\s*$", re.MULTILINE)


class Environment:
    """Holds the raw JD and slices it into focused snippets for leaf calls."""

    def __init__(self, jd_text: str):
        self.jd = (jd_text or "").strip()
        self._sections = self._split()

    def metadata(self) -> dict:
        return {"length": len(self.jd),
                "prefix": self.jd[:400],
                "sections_found": [k for k, v in self._sections.items() if v]}

    def _split(self) -> dict[str, str]:
        """Bucket the JD text under canonical sections by header lines.

        Best-effort: if no headers are recognized, every bucket falls back to the
        whole JD (via slice())."""
        buckets: dict[str, list[str]] = {k: [] for k in _SECTION_KEYS}
        headers = list(_HEADER_RE.finditer(self.jd))
        if not headers:
            return {k: "" for k in _SECTION_KEYS}
        spans = []
        for i, m in enumerate(headers):
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(self.jd)
            spans.append((m.group(1).lower(), self.jd[start:end].strip()))
        for htext, body in spans:
            for canon, keys in _SECTION_KEYS.items():
                if any(k in htext for k in keys):
                    buckets[canon].append(body)
                    break
        return {k: "\n".join(v).strip() for k, v in buckets.items()}

    def section(self, name: str) -> str:
        return self._sections.get(name, "") or ""

    def slice(self, *sections: str, head: int = 0) -> str:
        """Compose a snippet from named sections (+ optional JD head), capped.

        Falls back to the whole JD when the requested sections weren't detected —
        a short JD is fine to pass whole; the point is focus, not omission."""
        parts = []
        if head:
            parts.append(self.jd[:head])
        for s in sections:
            body = self.section(s)
            if body:
                parts.append(body)
        text = "\n\n".join(p for p in parts if p).strip()
        if len(text) < 120:          # detection missed → use the whole JD
            text = self.jd
        return text[:MAX_SNIPPET]


def llm_query(backend: Backend, prompt: str, *, retries: int = 1,
              temperature: float = 0.2, max_tokens: int = 2048) -> Any:
    """One leaf call: generate → tolerant JSON parse, with a light reformat retry.

    Returns the parsed JSON value, or None if the model never produced parseable
    JSON (the caller then applies a sentinel — we never crash on a bad leaf)."""
    p = prompt
    for attempt in range(retries + 1):
        try:
            text = backend.generate(p, system=SYSTEM, temperature=temperature,
                                    max_tokens=max_tokens)
            return extract_json(text)
        except Exception:  # noqa: BLE001 — parse or API hiccup; retry then give up
            if attempt >= retries:
                return None
            p = prompt + "\n\nReturn ONLY valid JSON, nothing else."
    return None
