"""Tolerant JSON extraction from LLM text output.

Small models wrap JSON in code fences, prose, or trailing commentary. These
helpers recover the intended JSON object/array without trusting the model to
emit clean output — the parsing discipline the RLM harness leans on.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|ya?ml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced open_ch..close_ch span (respecting strings)."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> Any:
    """Best-effort parse of the JSON value embedded in `text`.

    Tries, in order: the whole string, a fenced block, then the first balanced
    {..} or [..] span. Raises ValueError if nothing parses.
    """
    if text is None:
        raise ValueError("no text to parse")
    candidates: list[str] = []
    t = text.strip()
    candidates.append(t)
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    for oc, cc in (("{", "}"), ("[", "]")):
        span = _balanced(text, oc, cc)
        if span:
            candidates.append(span)
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"could not extract JSON from model output: {text[:200]!r}")
