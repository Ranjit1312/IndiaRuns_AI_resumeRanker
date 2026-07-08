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
from .candidate_fields import JsonSchemaValidator, ValidationResult, validate_candidate
from .rlm import ArtifactSpec, CompileOutcome, Environment, llm_query, run_compile


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
class ResumeSpec:
    """ArtifactSpec adapter for the candidate dict (Candidate A).

    Owns everything `compile_resume` used to inline: the ordered builder set,
    the assemble step (projects->summary fold, the faithfulness rule —
    projects text never enters career_history), and the sentinel map. Also
    runs the final forced-sentinel pass in `finalize`, mirroring the original
    "never emit an invalid candidate" last resort.
    """

    order = ["candidate_id", "profile", "career_history", "education", "skills",
             "redrob_signals"]

    def __init__(self, ref_date: "str | None" = None, schema_path: str = CF.SCHEMA_PATH):
        self.validator = JsonSchemaValidator(schema_path)
        self.ref_date = ref_date
        self._projects_text = ""   # captured during build(), folded in assemble()

    def build(self, name, env, backend, health, *, logger=None, on_event=None):
        tel = dict(logger=logger, on_event=on_event)
        if name == "candidate_id":
            return CF.new_candidate_id()
        if name == "profile":
            profile = build_profile(env, backend, health, **tel)
            self._projects_text = build_projects_summary(env, backend, health, **tel)
            return profile
        if name == "career_history":
            return build_career_history(env, backend, health, ref_date=self.ref_date, **tel)
        if name == "education":
            return build_education(env, backend, health, **tel)
        if name == "skills":
            return build_skills(env, backend, health, **tel)
        if name == "redrob_signals":
            return build_redrob_signals(env, backend, health, ref_date=self.ref_date)
        raise KeyError(name)   # pragma: no cover - exhaustive `order` above

    def assemble(self, parts: dict, env: Environment) -> dict:
        profile = CF.append_to_summary(parts["profile"], self._projects_text)
        return {
            "candidate_id": parts["candidate_id"],
            "profile": profile,
            "career_history": parts["career_history"],
            "education": parts["education"],
            "skills": parts["skills"],
            "redrob_signals": parts["redrob_signals"],
        }

    def rebuild(self, artifact, failing_top, env, backend, hint, health,
               *, logger=None, on_event=None):
        tel = dict(logger=logger, on_event=on_event)
        if failing_top == "profile":
            artifact["profile"] = build_profile(
                env, backend, health, hint=hint, leaf_prefix="repair.", **tel)
        elif failing_top == "career_history":
            artifact["career_history"] = build_career_history(
                env, backend, health, hint=hint, leaf_prefix="repair.",
                ref_date=self.ref_date, **tel)
        elif failing_top == "education":
            artifact["education"] = build_education(
                env, backend, health, hint=hint, leaf_prefix="repair.", **tel)
        elif failing_top == "skills":
            artifact["skills"] = build_skills(
                env, backend, health, hint=hint, leaf_prefix="repair.", **tel)
        elif failing_top == "redrob_signals":
            artifact["redrob_signals"] = build_redrob_signals(
                env, backend, health, ref_date=self.ref_date)
        elif failing_top == "candidate_id":
            artifact["candidate_id"] = CF.new_candidate_id()
        return artifact

    def sentinel(self, artifact, failing_top):
        s = _sentinel_for(failing_top, artifact, self.ref_date)
        if s is not None:
            artifact[failing_top] = s
        return artifact

    def finalize(self, artifact, env, backend, *, logger=None, on_event=None):
        """Last-resort guarantee: if still invalid after the repair budget,
        force every block through its sentinel coercion once (never emit an
        invalid candidate)."""
        if self.validator.validate(artifact).ok:
            return None
        artifact["candidate_id"] = artifact.get("candidate_id") or CF.new_candidate_id()
        artifact["profile"] = CF.coerce_profile(artifact.get("profile") or {})
        artifact["career_history"] = CF.coerce_career_history(
            artifact.get("career_history") or [], ref_date=self.ref_date)
        artifact["education"] = CF.coerce_education(artifact.get("education") or [])
        artifact["skills"] = CF.coerce_skills(artifact.get("skills") or [])
        artifact["redrob_signals"] = CF.coerce_redrob_signals(
            artifact.get("redrob_signals"), "", ref_date=self.ref_date)
        return "__final_pass__"


def compile_resume(resume_text: str, backend: Backend, *,
                   logger=None, on_event=None, ref_date: "str | None" = None,
                   max_repairs: int = 2) -> ResumeResult:
    env = Environment(resume_text)
    spec = ResumeSpec(ref_date=ref_date)

    outcome: CompileOutcome = run_compile(
        env, spec, backend, logger=logger, on_event=on_event, max_repairs=max_repairs)

    candidate = outcome.artifact
    health = outcome.health
    result = outcome.validation
    if outcome.extra == "__final_pass__":
        health["sentineled"].append("__final_pass__")
        result = spec.validator.validate(candidate)

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
