"""Field specs, sentinels, and coercion helpers for the JD seam.

Two jobs:
  1. Safe DEFAULTS for graceful degradation (intrinsic.py discipline) — a leaf
     that can't be extracted falls back here so we NEVER emit an invalid file.
  2. Small deterministic fixers (id slugging, regex safety, weight sanity) so a
     small model's near-miss output still assembles into a schema-valid profile.
"""
from __future__ import annotations

import re

# The engine implements exactly these four red-flag gates (method_config.damps +
# the stale_ic ladder). The harness may only TOGGLE them; other "do-not-want"
# intent goes to jd_meta.yaml / domain.out_of_domain_terms.
ALLOWED_RED_FLAGS = ["cv_primary", "job_hopper", "only_consulting", "stale_ic_role"]

DEFAULT_RED_FLAGS = {
    "cv_primary": {"enabled": False},
    "job_hopper": {"enabled": True},
    "only_consulting": {"enabled": False},
    "stale_ic_role": {"enabled": True},
}

DEFAULT_IDEAL_EXPERIENCE = {
    "min_years": 2, "max_years": 10, "peak_years": 5.0, "sigma_years": 2.5}

DEFAULT_LOCATIONS = {
    "preferred": [], "acceptable": [],
    "relocation_acceptable": True, "remote_acceptable": False}

DEFAULT_DENSE_EXTRAS = {"yoe_fit_weight": 0.08, "domain_ratio_weight": 0.10}

DEFAULT_DOMAIN = {
    "in_domain_terms": ["engineering", "technology"],
    "out_of_domain_terms": ["unrelated field"],
    "in_domain_regex": r"engineer|technolog|software|data|product",
    "out_of_domain_regex": r"unrelated field",
}

DEFAULT_RELEVANT_SKILL_REGEX = (
    r"python|java|sql|cloud|aws|azure|gcp|communication|leadership|analysis")

# leaf task -> the jd_profile keys it fills (documents the decomposition)
LEAF_TASKS = [
    "role", "locations", "signals", "domain",
    "relevant_skill_regex", "red_flags", "cross_encoder_query",
]


def slugify_id(text: str, taken: set[str]) -> str:
    """A schema-legal signal id: ^[a-z][a-z0-9_]*$, deduped."""
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "sig_" + s if s else "signal"
    s = s[:32].rstrip("_") or "signal"
    base, i = s, 2
    while s in taken:
        s = f"{base}_{i}"
        i += 1
    taken.add(s)
    return s


def safe_regex(pattern) -> str | None:
    """Return the pattern if it compiles (case-insensitive), else None.

    A bad evidence_regex must not sink the whole profile — null is a valid,
    documented value (a model-only signal axis)."""
    if pattern is None:
        return None
    if not isinstance(pattern, str) or not pattern.strip():
        return None
    try:
        re.compile(pattern, re.I)
        return pattern
    except re.error:
        return None


def as_number(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def sanitize_red_flags(rf) -> dict:
    """Keep only the 4 supported gates; coerce to {enabled: bool}."""
    out = {}
    src = rf if isinstance(rf, dict) else {}
    for name in ALLOWED_RED_FLAGS:
        body = src.get(name)
        if isinstance(body, dict) and "enabled" in body:
            out[name] = {"enabled": bool(body["enabled"])}
        elif isinstance(body, bool):
            out[name] = {"enabled": body}
        else:
            out[name] = dict(DEFAULT_RED_FLAGS[name])
    return out


def sanitize_signals(signals) -> list[dict]:
    """Force ids legal/unique, drop invalid regexes to null, coerce weights."""
    out, taken = [], set()
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        label = str(s.get("label") or s.get("id") or "signal").strip()
        sid = s.get("id")
        if not (isinstance(sid, str) and re.match(r"^[a-z][a-z0-9_]*$", sid)):
            sid = slugify_id(sid or label, taken)
        else:
            if sid in taken:
                sid = slugify_id(sid, taken)
            else:
                taken.add(sid)
        query = str(s.get("query") or label).strip() or label
        out.append({
            "id": sid,
            "label": label,
            "query": query,
            "evidence_regex": safe_regex(s.get("evidence_regex")),
            "dense_weight": max(0.0, as_number(s.get("dense_weight"), 0.1)),
        })
    return out
