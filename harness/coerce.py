"""compile_jd — the RLM root plan: leaves → assemble → validate → repair → emit.

Deterministic root (this code) dispatches focused leaf extractions, assembles a
jd_profile dict with defensive sanitizers, validates via the engine's
profile.load, and repairs ONLY the failing block (one model re-call, then a
guaranteed-valid sentinel). Also emits the coaching sidecar jd_meta. Never
returns an invalid profile.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import yaml

from . import prompts as P
from . import schema_fields as F
from .backends import Backend, DEFAULT_MODEL
from .rlm import Environment, llm_query
from .validate import METHOD_PATH, ValidationResult, validate_profile_dict

MAX_SIGNALS = 7


@dataclass
class CompileResult:
    profile: dict
    meta: dict
    health: dict
    validation: ValidationResult
    profile_yaml: str
    meta_yaml: str


# --------------------------------------------------------------------------- #
# per-block builders (each defensive: bad/empty leaf → sentinel, logged)
# --------------------------------------------------------------------------- #
def build_role(env: Environment, backend: Backend, health: dict, hint: str = "") -> dict:
    snip = env.slice("qualifications", head=1400)
    d = llm_query(backend, P.role_prompt(snip) + _hint(hint)) or {}
    if not isinstance(d, dict):
        d = {}
    title = str(d.get("title") or _first_line(env.jd) or "Role").strip()
    ie = {
        "min_years": F.as_number(d.get("min_years"), F.DEFAULT_IDEAL_EXPERIENCE["min_years"]),
        "max_years": F.as_number(d.get("max_years"), F.DEFAULT_IDEAL_EXPERIENCE["max_years"]),
        "peak_years": F.as_number(d.get("peak_years"), F.DEFAULT_IDEAL_EXPERIENCE["peak_years"]),
        "sigma_years": max(1.0, F.as_number(d.get("sigma_years"), 2.5)),
    }
    if not d:
        health["defaulted"].append("role")
    return {
        "title": title,
        "company": str(d.get("company") or "").strip(),
        "domain": str(d.get("domain") or "general").strip() or "general",
        "summary": str(d.get("summary") or "").strip(),
        "ideal_experience": ie,
        "notice_preference_days": int(F.as_number(d.get("notice_preference_days"), 60)),
    }


def build_locations(env: Environment, backend: Backend, health: dict, hint: str = "") -> dict:
    snip = env.slice("location", "about", head=600)
    d = llm_query(backend, P.locations_prompt(snip) + _hint(hint))
    if not isinstance(d, dict):
        health["defaulted"].append("locations")
        return dict(F.DEFAULT_LOCATIONS)
    return {
        "preferred": _lc_list(d.get("preferred")),
        "acceptable": _lc_list(d.get("acceptable")),
        "relocation_acceptable": bool(d.get("relocation_acceptable", True)),
        "remote_acceptable": bool(d.get("remote_acceptable", False)),
    }


def build_signals(env: Environment, backend: Backend, health: dict, hint: str = "") -> list[dict]:
    snip = env.slice("responsibilities", "qualifications")
    labels = llm_query(backend, P.signal_labels_prompt(snip) + _hint(hint))
    if not isinstance(labels, list) or not labels:
        health["defaulted"].append("signals")
        return F.sanitize_signals([{
            "id": "core_fit", "label": "core role fit",
            "query": snip[:160] or "core responsibilities of the role",
            "evidence_regex": None, "dense_weight": 0.2}])
    raw = []
    for label in [str(x) for x in labels][:MAX_SIGNALS]:
        det = llm_query(backend, P.signal_detail_prompt(label, snip)) or {}
        det = det if isinstance(det, dict) else {}
        raw.append({"id": label, "label": label,
                    "query": det.get("query") or label,
                    "evidence_regex": det.get("evidence_regex"),
                    "dense_weight": det.get("dense_weight", 0.15)})
    sigs = F.sanitize_signals(raw)
    return sigs or F.sanitize_signals([{"id": "core_fit", "label": "core role fit",
                                        "query": snip[:160], "evidence_regex": None,
                                        "dense_weight": 0.2}])


def build_domain(env: Environment, backend: Backend, health: dict, hint: str = "") -> dict:
    snip = env.slice("responsibilities", "qualifications")
    d = llm_query(backend, P.domain_prompt(snip) + _hint(hint))
    if not isinstance(d, dict):
        health["defaulted"].append("domain")
        return dict(F.DEFAULT_DOMAIN)
    in_terms = _lc_list(d.get("in_domain_terms")) or F.DEFAULT_DOMAIN["in_domain_terms"]
    out_terms = _lc_list(d.get("out_of_domain_terms")) or F.DEFAULT_DOMAIN["out_of_domain_terms"]
    return {
        "in_domain_terms": in_terms,
        "out_of_domain_terms": out_terms,
        "in_domain_regex": F.safe_regex(d.get("in_domain_regex")) or _terms_regex(in_terms),
        "out_of_domain_regex": F.safe_regex(d.get("out_of_domain_regex")) or _terms_regex(out_terms),
    }


def build_relevant_skill(env: Environment, backend: Backend, health: dict, hint: str = "") -> str:
    snip = env.slice("qualifications", "responsibilities")
    d = llm_query(backend, P.relevant_skill_prompt(snip) + _hint(hint))
    rx = F.safe_regex((d or {}).get("relevant_skill_regex") if isinstance(d, dict) else None)
    if not rx:
        health["defaulted"].append("relevant_skill_regex")
        return F.DEFAULT_RELEVANT_SKILL_REGEX
    return rx


def build_red_flags(env: Environment, backend: Backend, health: dict, hint: str = "") -> dict:
    d = llm_query(backend, P.red_flags_prompt(env.slice(head=2500)) + _hint(hint))
    if not isinstance(d, dict):
        health["defaulted"].append("red_flags")
    return F.sanitize_red_flags(d)


def build_cross_encoder(role: dict, signals: list[dict]) -> str:
    """Deterministic positive paraphrase — no extra model call needed."""
    caps = ", ".join(s["label"] for s in signals[:5])
    base = role.get("summary") or role.get("title") or "the ideal candidate"
    return (f"{base} — a strong fit demonstrates {caps}, with relevant hands-on "
            f"experience around {role.get('domain','the domain')}.").strip()


# --------------------------------------------------------------------------- #
# root plan
# --------------------------------------------------------------------------- #
_BUILDERS = {
    "role": build_role, "locations": build_locations, "signals": build_signals,
    "domain": build_domain, "relevant_skill_regex": build_relevant_skill,
    "red_flags": build_red_flags,
}

_SENTINELS = {
    "locations": lambda prof: dict(F.DEFAULT_LOCATIONS),
    "domain": lambda prof: dict(F.DEFAULT_DOMAIN),
    "relevant_skill_regex": lambda prof: F.DEFAULT_RELEVANT_SKILL_REGEX,
    "red_flags": lambda prof: {k: dict(v) for k, v in F.DEFAULT_RED_FLAGS.items()},
    "signals": lambda prof: F.sanitize_signals([{
        "id": "core_fit", "label": "core role fit",
        "query": prof.get("role", {}).get("summary") or "core responsibilities",
        "evidence_regex": None, "dense_weight": 0.2}]),
    "role": lambda prof: {**prof.get("role", {}),
                          "title": prof.get("role", {}).get("title") or "Role",
                          "domain": prof.get("role", {}).get("domain") or "general",
                          "ideal_experience": dict(F.DEFAULT_IDEAL_EXPERIENCE),
                          "notice_preference_days": 60},
}


def compile_jd(jd_text: str, backend: Backend, *, method_path: str = METHOD_PATH,
               max_repairs: int = 2, source: dict | None = None) -> CompileResult:
    env = Environment(jd_text)
    health = {"defaulted": [], "sentineled": [], "repairs": 0,
              "metadata": env.metadata(), "model": getattr(backend, "name", "?")}

    role = build_role(env, backend, health)
    signals = build_signals(env, backend, health)
    profile = {
        "schema_version": 1,
        "role": role,
        "locations": build_locations(env, backend, health),
        "signals": signals,
        "dense_extras": dict(F.DEFAULT_DENSE_EXTRAS),
        "cross_encoder_query": build_cross_encoder(role, signals),
        "domain": build_domain(env, backend, health),
        "relevant_skill_regex": build_relevant_skill(env, backend, health),
        "red_flags": build_red_flags(env, backend, health),
    }

    # validate → repair only the failing block (re-call once, then sentinel)
    result = validate_profile_dict(profile, method_path)
    while not result.ok and health["repairs"] < max_repairs:
        health["repairs"] += 1
        top = result.top
        if top in _BUILDERS:
            try:
                profile[top] = _BUILDERS[top](env, backend, health, hint=result.error or "")
                if top == "role":  # ce query mentions role fields
                    profile["cross_encoder_query"] = build_cross_encoder(profile["role"], profile["signals"])
            except Exception:  # noqa: BLE001
                pass
            result = validate_profile_dict(profile, method_path)
        if not result.ok and top in _SENTINELS:
            profile[top] = _SENTINELS[top](profile)
            health["sentineled"].append(top)
            result = validate_profile_dict(profile, method_path)
        elif top not in _SENTINELS and top not in _BUILDERS:
            break   # unlocatable field — stop rather than loop

    meta = build_meta(env, backend, profile, source or {})
    return CompileResult(
        profile=profile, meta=meta, health=health, validation=result,
        profile_yaml=to_yaml(profile), meta_yaml=to_yaml(meta))


def build_meta(env: Environment, backend: Backend, profile: dict, source: dict) -> dict:
    d = llm_query(backend, P.meta_extras_prompt(env.slice("qualifications", "responsibilities", head=1200)))
    d = d if isinstance(d, dict) else {}
    sigs = profile.get("signals", [])
    out_terms = set(profile.get("domain", {}).get("out_of_domain_terms", []))

    def enforced(text: str) -> str:
        t = text.lower()
        return "domain" if any(term in t for term in out_terms) else "none"

    return {
        "source": source or {"note": "provided at compile time"},
        "hard_requirements": [
            {"text": str(r.get("text", "")).strip(),
             "kind": r.get("kind", "skill"),
             "enforced_by": "yoe" if r.get("kind") == "yoe"
                            else "location" if r.get("kind") == "location"
                            else "none"}
            for r in (d.get("hard_requirements") or []) if isinstance(r, dict)],
        "must_haves": [{"text": s["label"], "signal_id": s["id"]}
                       for s in sorted(sigs, key=lambda s: -s["dense_weight"])[:4]],
        "nice_to_haves": [{"text": s["label"], "signal_id": s["id"]}
                          for s in sorted(sigs, key=lambda s: -s["dense_weight"])[4:]],
        "explicit_exclusions": [
            {"text": str(x.get("text", "")).strip(), "enforced_by": enforced(str(x.get("text", ""))),
             "note": ""} for x in (d.get("explicit_exclusions") or []) if isinstance(x, dict)],
        "coaching_notes": str(d.get("coaching_notes") or "").strip(),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hint(hint: str) -> str:
    return f"\n\nA previous attempt failed validation: {hint}\nFix that and return valid JSON." if hint else ""


def _first_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()[:80]
    return ""


def _lc_list(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip().lower() for x in v if str(x).strip()]


def _terms_regex(terms: list[str]) -> str:
    import re as _re
    esc = [_re.escape(t) for t in terms if t]
    return "|".join(esc) if esc else "unrelated field"


def to_yaml(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- #
# CLI: python -m harness.coerce --jd data/eval_jds/<slug>/jd.txt
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile a JD into jd_profile.yaml + jd_meta.yaml")
    ap.add_argument("--jd", required=True, help="path to a JD text file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="dir to write jd_profile.yaml + jd_meta.yaml")
    args = ap.parse_args(argv)

    from .backends import make_backend
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GOOGLE_API_KEY to run the live harness.")
        return 2
    jd = open(args.jd, encoding="utf-8").read()
    res = compile_jd(jd, make_backend(key, args.model))
    print(f"[compile] model={args.model} valid={res.validation.ok} "
          f"repairs={res.health['repairs']} defaulted={res.health['defaulted']} "
          f"sentineled={res.health['sentineled']}")
    if not res.validation.ok:
        print(f"[compile] STILL INVALID: {res.validation.error}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        open(os.path.join(args.out, "jd_profile.yaml"), "w", encoding="utf-8").write(res.profile_yaml)
        open(os.path.join(args.out, "jd_meta.yaml"), "w", encoding="utf-8").write(res.meta_yaml)
        print(f"[compile] wrote to {args.out}")
    else:
        print(res.profile_yaml)
    return 0 if res.validation.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
