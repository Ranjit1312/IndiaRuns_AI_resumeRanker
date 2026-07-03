"""compile_resume — the RLM root plan for resume->candidate (Phase 2, Part C2).

Mirrors `harness/coerce.py`'s discipline: a deterministic root (this code)
dispatches focused leaf extractions over resume text, assembles a candidate
dict with defensive sanitizers (`harness.candidate_fields`), validates via
the authoritative `candidate_schema.json` (`validate_candidate`), and repairs
ONLY the failing top-level block (one bounded model re-call, then a
guaranteed-valid sentinel coercion). Never returns an invalid candidate.

Faithfulness rule (docs/PHASE2_SPEC.md): standalone projects are appended to
`profile.summary`, NOT `career_history` — folding them into career_history
would distort the engine's tenure/hopper/integrity math in
`redrob_ranker/fit.py` (avg_tenure_months, n_jobs, max_role_months all read
career_history verbatim).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import yaml

from . import candidate_fields as CF
from . import resume_prompts as RP
from .backends import Backend, DEFAULT_MODEL
from .candidate_fields import ValidationResult, validate_candidate
from .rlm import Environment, llm_query


@dataclass
class ResumeResult:
    candidate: dict
    health: dict
    validation: ValidationResult
    candidate_yaml: str
    candidate_json: str


# --------------------------------------------------------------------------- #
# per-block builders (each defensive: bad/empty leaf -> sentinel, logged)
# --------------------------------------------------------------------------- #
def build_profile(env: Environment, backend: Backend, health: dict, hint: str = "",
                  *, logger=None, on_event=None, leaf_prefix: str = "") -> dict:
    snip = env.slice(head=2500)
    d = llm_query(backend, RP.profile_prompt(snip) + _hint(hint),
                  leaf=f"{leaf_prefix}profile", logger=logger, on_event=on_event)
    if not isinstance(d, dict) or not d:
        health["defaulted"].append("profile")
        d = {}
    return CF.coerce_profile(d)


def build_career_history(env: Environment, backend: Backend, health: dict, hint: str = "",
                         *, logger=None, on_event=None, leaf_prefix: str = "",
                         ref_date: "str | None" = None) -> list:
    snip = env.slice(head=5000)
    d = llm_query(backend, RP.career_history_prompt(snip) + _hint(hint),
                  leaf=f"{leaf_prefix}career_history", logger=logger, on_event=on_event)
    if not isinstance(d, list) or not d:
        health["defaulted"].append("career_history")
        d = []
    return CF.coerce_career_history(d, ref_date=ref_date)


def build_education(env: Environment, backend: Backend, health: dict, hint: str = "",
                    *, logger=None, on_event=None, leaf_prefix: str = "") -> list:
    snip = env.slice(head=3000)
    d = llm_query(backend, RP.education_prompt(snip) + _hint(hint),
                  leaf=f"{leaf_prefix}education", logger=logger, on_event=on_event)
    if not isinstance(d, list):
        health["defaulted"].append("education")
        d = []
    return CF.coerce_education(d)


def build_skills(env: Environment, backend: Backend, health: dict, hint: str = "",
                 *, logger=None, on_event=None, leaf_prefix: str = "") -> list:
    snip = env.slice(head=4000)
    d = llm_query(backend, RP.skills_prompt(snip) + _hint(hint),
                  leaf=f"{leaf_prefix}skills", logger=logger, on_event=on_event)
    if not isinstance(d, list):
        health["defaulted"].append("skills")
        d = []
    return CF.coerce_skills(d)


def build_projects_summary(env: Environment, backend: Backend, health: dict,
                           *, logger=None, on_event=None, leaf_prefix: str = "") -> str:
    """Projects -> plain text ONLY, folded into profile.summary by the root
    (never career_history — see module docstring's faithfulness rule)."""
    snip = env.slice(head=4000)
    d = llm_query(backend, RP.projects_prompt(snip),
                  leaf=f"{leaf_prefix}projects", logger=logger, on_event=on_event)
    if not isinstance(d, dict):
        health["defaulted"].append("projects")
        return ""
    text = str(d.get("projects_summary") or "").strip()
    if text:
        text = f"Projects: {text}"
    return text


def build_redrob_signals(env: Environment, backend: Backend, health: dict,
                         *, ref_date: "str | None" = None) -> dict:
    """No LLM call needed for most of the 23 signals (a resume can't supply
    platform-activity data) — deterministic sentinel + resume-text hints."""
    return CF.coerce_redrob_signals(None, env.jd, ref_date=ref_date)


# --------------------------------------------------------------------------- #
# sentinel repair map — guaranteed-valid fallback per top-level block
# --------------------------------------------------------------------------- #
def _sentinel_for(top: "str | None", cand: dict, ref_date: "str | None") -> "object | None":
    if top == "profile":
        return CF.coerce_profile(cand.get("profile") or {})
    if top == "career_history":
        return CF.coerce_career_history(cand.get("career_history") or [], ref_date=ref_date)
    if top == "education":
        return CF.coerce_education(cand.get("education") or [])
    if top == "skills":
        return CF.coerce_skills(cand.get("skills") or [])
    if top == "redrob_signals":
        return CF.coerce_redrob_signals(cand.get("redrob_signals"), "", ref_date=ref_date)
    if top == "candidate_id":
        return CF.new_candidate_id()
    return None


# --------------------------------------------------------------------------- #
# root plan
# --------------------------------------------------------------------------- #
def compile_resume(resume_text: str, backend: Backend, *,
                   logger=None, on_event=None, ref_date: "str | None" = None,
                   max_repairs: int = 2) -> ResumeResult:
    env = Environment(resume_text)
    health = {"defaulted": [], "sentineled": [], "repairs": 0,
              "metadata": env.metadata(), "model": getattr(backend, "name", "?")}
    tel = dict(logger=logger, on_event=on_event)   # threaded into every leaf call

    profile = build_profile(env, backend, health, **tel)
    career_history = build_career_history(env, backend, health, ref_date=ref_date, **tel)
    education = build_education(env, backend, health, **tel)
    skills = build_skills(env, backend, health, **tel)
    projects_text = build_projects_summary(env, backend, health, **tel)
    profile = CF.append_to_summary(profile, projects_text)
    redrob_signals = build_redrob_signals(env, backend, health, ref_date=ref_date)

    candidate = {
        "candidate_id": CF.new_candidate_id(),
        "profile": profile,
        "career_history": career_history,
        "education": education,
        "skills": skills,
        "redrob_signals": redrob_signals,
    }

    # validate -> repair only the failing block (one bounded re-call, then a
    # guaranteed-valid sentinel — mirrors compile_jd's loop exactly).
    result = validate_candidate(candidate)
    while not result.ok and health["repairs"] < max_repairs:
        health["repairs"] += 1
        top = result.top
        try:
            if top == "profile":
                candidate["profile"] = build_profile(
                    env, backend, health, hint=result.error or "",
                    leaf_prefix="repair.", **tel)
            elif top == "career_history":
                candidate["career_history"] = build_career_history(
                    env, backend, health, hint=result.error or "",
                    leaf_prefix="repair.", ref_date=ref_date, **tel)
            elif top == "education":
                candidate["education"] = build_education(
                    env, backend, health, hint=result.error or "",
                    leaf_prefix="repair.", **tel)
            elif top == "skills":
                candidate["skills"] = build_skills(
                    env, backend, health, hint=result.error or "",
                    leaf_prefix="repair.", **tel)
            elif top == "redrob_signals":
                candidate["redrob_signals"] = build_redrob_signals(
                    env, backend, health, ref_date=ref_date)
            elif top == "candidate_id":
                candidate["candidate_id"] = CF.new_candidate_id()
        except Exception:  # noqa: BLE001 — a leaf hiccup must not sink compile
            pass
        result = validate_candidate(candidate)

        if not result.ok:
            sentinel = _sentinel_for(top, candidate, ref_date)
            if sentinel is not None:
                candidate[top] = sentinel
                health["sentineled"].append(top)
                result = validate_candidate(candidate)
            elif top not in {"profile", "career_history", "education", "skills",
                             "redrob_signals", "candidate_id"}:
                break   # unlocatable field — stop rather than loop

    # last-resort guarantee: if still invalid after the repair budget, force
    # every block through its sentinel coercion once (never emit an invalid
    # candidate — mirrors compile_jd's contract).
    if not result.ok:
        candidate["candidate_id"] = candidate.get("candidate_id") or CF.new_candidate_id()
        candidate["profile"] = CF.coerce_profile(candidate.get("profile") or {})
        candidate["career_history"] = CF.coerce_career_history(
            candidate.get("career_history") or [], ref_date=ref_date)
        candidate["education"] = CF.coerce_education(candidate.get("education") or [])
        candidate["skills"] = CF.coerce_skills(candidate.get("skills") or [])
        candidate["redrob_signals"] = CF.coerce_redrob_signals(
            candidate.get("redrob_signals"), "", ref_date=ref_date)
        health["sentineled"].append("__final_pass__")
        result = validate_candidate(candidate)

    if logger is not None:
        health["telemetry"] = logger.summary()

    return ResumeResult(
        candidate=candidate, health=health, validation=result,
        candidate_yaml=to_yaml(candidate),
        candidate_json=to_json(candidate),
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hint(hint: str) -> str:
    return f"\n\nA previous attempt failed validation: {hint}\nFix that and return valid JSON." \
        if hint else ""


def to_yaml(d: dict) -> str:
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100)


def to_json(d: dict) -> str:
    import json
    return json.dumps(d, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# CLI: python -m harness.resume --resume path/to/resume.txt
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile a resume into a candidate.yaml")
    ap.add_argument("--resume", required=True, help="path to a resume text file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="path to write candidate.yaml")
    ap.add_argument("--ref-date", default=None)
    args = ap.parse_args(argv)

    from .backends import make_backend
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GOOGLE_API_KEY to run the live harness.")
        return 2
    text = open(args.resume, encoding="utf-8").read()
    res = compile_resume(text, make_backend(key, args.model), ref_date=args.ref_date)
    print(f"[resume] model={args.model} valid={res.validation.ok} "
          f"repairs={res.health['repairs']} defaulted={res.health['defaulted']} "
          f"sentineled={res.health['sentineled']}")
    if not res.validation.ok:
        print(f"[resume] STILL INVALID: {res.validation.error}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(res.candidate_yaml)
        print(f"[resume] wrote to {args.out}")
    else:
        print(res.candidate_yaml)
    return 0 if res.validation.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
