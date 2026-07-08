"""Depth-1 RLM engine: the JD as an external variable + focused leaf calls.

Faithful to arXiv 2512.24601 at the scale that matters for a short JD: the input
lives as a variable the *root* (our deterministic code) inspects and slices, and
work is delegated to leaf `llm_query` calls that each see only a relevant snippet
+ one sub-schema. The paper's lesson — small models win by learning *when/how to
delegate*, not by leaf reasoning — is why the root is code, not the model.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .backends import Backend
from .jsonutil import extract_json
from .logging_utils import LeafEntry
from .prompts import SYSTEM
from .validate import Validator, ValidationResult

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


def llm_query(backend: Backend, prompt: str, *, leaf: str = "", logger=None,
              on_event=None, retries: int = 1, temperature: float = 0.2,
              max_tokens: int = 2048) -> Any:
    """One leaf call: generate → tolerant JSON parse, with a light reformat retry.

    Returns the parsed JSON value, or None if the model never produced parseable
    JSON (the caller then applies a sentinel — we never crash on a bad leaf).

    When `logger` and/or `on_event` are supplied, records the call's prompt,
    system, raw response, parsed value, elapsed time, token usage, and ok/error
    for live telemetry. Never raises for parse/API errors (including
    RateLimitError, which is a RuntimeError caught below); those surface only via
    the recorded entry's `error` field so the sentinel path keeps working."""
    p = prompt
    t0 = time.monotonic()
    raw_response: str | None = None
    parsed: Any = None
    error: str | None = None
    for attempt in range(retries + 1):
        try:
            raw_response = backend.generate(p, system=SYSTEM, temperature=temperature,
                                            max_tokens=max_tokens)
            parsed = extract_json(raw_response)
            if parsed is not None:
                break
        except Exception as exc:  # noqa: BLE001 — parse or API hiccup; retry then give up
            error = str(exc)
            raw_response = None
        if attempt >= retries:
            break
        p = prompt + "\n\nReturn ONLY valid JSON, nothing else."

    elapsed_s = time.monotonic() - t0
    ok = parsed is not None
    usage = getattr(backend, "last_usage", None)
    if logger is not None:
        logger.record(leaf=leaf, model=getattr(backend, "name", "?"), prompt=prompt,
                      system=SYSTEM, raw_response=raw_response, parsed=parsed,
                      elapsed_s=elapsed_s, ok=ok, error=error, tokens=usage)
    if on_event is not None:
        entry = logger.entries[-1] if logger is not None else LeafEntry(
            leaf=leaf, model=getattr(backend, "name", "?"), prompt=prompt,
            system=SYSTEM, raw_response=raw_response, parsed=parsed,
            elapsed_s=elapsed_s, ok=ok, error=error, tokens=usage)
        on_event(entry)
    return parsed


# --------------------------------------------------------------------------- #
# Compile loop (Candidate A) — the shared "never emit an invalid artifact"
# invariant that both compile_jd and compile_resume implement identically:
# build ordered leaves -> assemble -> validate -> repair-the-failing-block ->
# sentinel-degrade -> re-validate -> optional finalize.
# --------------------------------------------------------------------------- #
@runtime_checkable
class ArtifactSpec(Protocol):
    """What `run_compile` needs to compile one artifact type.

    Two adapters live today: `harness.coerce.JDSpec` (jd_profile.yaml, via
    `EngineProfileValidator`) and `harness.resume.ResumeSpec` (candidate dict,
    via `JsonSchemaValidator`).
    """

    #: leaf names, in build order (also the vocabulary `rebuild`/`sentinel`
    #: are keyed on, i.e. the artifact's top-level blocks).
    order: list[str]

    #: the validator this spec's artifact must satisfy.
    validator: Validator

    def build(self, name: str, env: "Environment", backend: Backend, health: dict,
             *, logger=None, on_event=None) -> Any:
        """Build one leaf/top-level block, in isolation."""
        ...

    def assemble(self, parts: dict[str, Any], env: "Environment") -> dict:
        """Combine the built parts into the artifact dict (derived fields,
        folds, etc. live here — e.g. JD's cross_encoder_query, résumé's
        projects->summary fold)."""
        ...

    def rebuild(self, artifact: dict, failing_top: "str | None", env: "Environment",
               backend: Backend, hint: str, health: dict,
               *, logger=None, on_event=None) -> dict:
        """Re-derive just the failing top-level block (one bounded model
        re-call). May also touch other derived fields as a special case
        (e.g. JD refreshing cross_encoder_query when role repairs). Returns
        the (possibly mutated) artifact dict."""
        ...

    def sentinel(self, artifact: dict, failing_top: "str | None") -> dict:
        """Guaranteed-valid fallback coercion for the failing top-level
        block. Returns the (possibly mutated) artifact dict."""
        ...

    def finalize(self, artifact: dict, env: "Environment", backend: Backend,
                *, logger=None, on_event=None) -> Any:
        """Optional post-validate step (e.g. JD's jd_meta sidecar). Specs
        without one simply omit the method — `run_compile` checks with
        `getattr`."""
        ...


@dataclass
class CompileOutcome:
    artifact: dict
    extra: Any               # finalize()'s return value, or None
    health: dict
    validation: ValidationResult


def run_compile(env: "Environment", spec: ArtifactSpec, backend: Backend, *,
                logger=None, on_event=None, max_repairs: int = 2) -> CompileOutcome:
    """The one deep module owning "never emit an invalid artifact":

    build ordered leaves -> spec.assemble(parts, env) -> validate via
    spec.validator -> while invalid and repairs<max: spec.rebuild(...) else
    spec.sentinel(...) -> re-validate -> optional spec.finalize(...).

    Threads logger/on_event/health exactly as compile_jd did before the
    refactor; both compile_jd and compile_resume are now thin specs over
    this loop.
    """
    health = {"defaulted": [], "sentineled": [], "repairs": 0,
              "metadata": env.metadata(), "model": getattr(backend, "name", "?")}
    tel = dict(logger=logger, on_event=on_event)

    parts = {name: spec.build(name, env, backend, health, **tel) for name in spec.order}
    artifact = spec.assemble(parts, env)

    result = spec.validator.validate(artifact)
    while not result.ok and health["repairs"] < max_repairs:
        health["repairs"] += 1
        top = result.top
        if top in spec.order:
            try:
                artifact = spec.rebuild(artifact, top, env, backend, result.error or "",
                                        health, **tel)
            except Exception:  # noqa: BLE001 — a leaf hiccup must not sink compile
                pass
            result = spec.validator.validate(artifact)
        if not result.ok and top in spec.order:
            artifact = spec.sentinel(artifact, top)
            health["sentineled"].append(top)
            result = spec.validator.validate(artifact)
        elif top not in spec.order:
            break   # unlocatable field — stop rather than loop

    extra = None
    finalize = getattr(spec, "finalize", None)
    if finalize is not None:
        extra = finalize(artifact, env, backend, **tel)

    if logger is not None:
        health["telemetry"] = logger.summary()

    return CompileOutcome(artifact=artifact, extra=extra, health=health, validation=result)
