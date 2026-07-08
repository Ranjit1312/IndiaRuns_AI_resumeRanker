"""fit_session — the résumé→fit orchestration seam ("Fit session" in CONTEXT.md).

Owns the lifecycle of turning one résumé into a persisted fit run:

    parse_resume  : résumé text -> ParsedResume            (pre-HITL)
    (human correction happens between the two calls, outside this module)
    run_fit       : candidate + JD profile -> FitOutcome    (validate -> load ->
                    score -> persist)

Both single-résumé and batch UI flows call the same two functions so they
persist identically (resume + candidate_record + fit_run) and render off the
same `FitOutcome.status` instead of each re-implementing try/except around
`RateLimitError`/`TransientBackendError`/`validate_candidate`.

No exceptions escape for the four known failure modes (ok/rate_limited/
transient/invalid); anything else is a genuine bug and propagates.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field

import yaml

from harness.backends import Backend, RateLimitError, TransientBackendError
from harness.candidate_fields import ValidationResult, validate_candidate
from harness.logging_utils import HarnessLogger
from harness.resume import compile_resume
from redrob_ranker import profile as rprofile
from redrob_ranker.fit import FitResult, score_candidate
from store.base import Workspace
from store.schema import CandidateRecordRow, FitRunRow, ProfileRow, ResumeRow

_HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_PATH = os.path.join(_HERE, "jd", "method_config.yaml")


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #
@dataclass
class ParsedResume:
    """Everything `parse_resume` produces, before human correction."""
    candidate: dict
    health: dict
    validation: ValidationResult
    telemetry: list = field(default_factory=list)   # HarnessLogger entries, as dicts


@dataclass
class FitOutcome:
    """The typed result of `run_fit`. UI renders off `status`, never exceptions."""
    status: str                      # "ok" | "rate_limited" | "transient" | "invalid"
    fit: "FitResult | None"
    fit_run_id: "str | None"
    message: str
    telemetry: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small id/time helpers (mirrors app.py's _now_iso / _new_id)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------- #
# parse_resume — everything before the HITL form
# --------------------------------------------------------------------------- #
def parse_resume(text: str, backend: Backend, *, on_event=None,
                  ref_date: "str | None" = None, max_repairs: int = 2) -> ParsedResume:
    """Compile résumé text into a `ParsedResume` (RLM harness + telemetry).

    Owns `compile_resume`'s logger wiring so callers don't build/thread a
    `HarnessLogger` themselves. No persistence here — that's `run_fit`'s job,
    after the HITL correction step (single-résumé flow) or immediately
    (batch flow, which auto-approves).
    """
    logger = HarnessLogger()
    result = compile_resume(text, backend, logger=logger, on_event=on_event,
                             ref_date=ref_date, max_repairs=max_repairs)
    return ParsedResume(
        candidate=result.candidate,
        health=result.health,
        validation=result.validation,
        telemetry=[e.as_dict() for e in logger.entries],
    )


# --------------------------------------------------------------------------- #
# JD profile loading — mirrors harness/validate.py's temp-file + rprofile.load
# pattern (kept local to fit_session, NOT imported from app.py)
# --------------------------------------------------------------------------- #
def _load_profile_method_from_yaml(profile_yaml_text: str, method_path: str = METHOD_PATH):
    """Load (Profile, Method) objects from an in-session jd_profile.yaml STRING,
    by writing it to a temp file — the engine's loader (`redrob_ranker.profile.
    load`) only reads from disk. Mirrors `harness/validate.py:validate_profile_
    dict`'s temp-file pattern; not cached here (callers that want per-session
    caching, e.g. Streamlit's `st.cache_resource`, wrap this themselves)."""
    d = yaml.safe_load(profile_yaml_text)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(d, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()
        return rprofile.load(tmp.name, method_path)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# run_fit — everything after HITL approval: validate -> load -> score -> persist
# --------------------------------------------------------------------------- #
def run_fit(candidate: dict, jd_profile_yaml: str, backend: Backend, store: Workspace,
            *, workspace_id: str, on_event=None, ref_date: "str | None" = None,
            jd_label: str = "JD", resume_text: str = "",
            resume_name: "str | None" = None) -> FitOutcome:
    """Validate -> load JD profile/method -> score -> persist (resume +
    candidate_record + fit_run). Returns a `FitOutcome`; never raises for the
    four known failure modes.

    Persistence is unified here so single-résumé and batch flows save
    identically: both write a `resume` row, a `candidate_record` row, a
    `profile` row (the JD), and a `fit_run` row.
    """
    vr = validate_candidate(candidate)
    if not vr.ok:
        return FitOutcome(status="invalid", fit=None, fit_run_id=None,
                          message=f"Candidate failed validation: {vr.error}",
                          telemetry=[])

    try:
        jd_profile, jd_method = _load_profile_method_from_yaml(jd_profile_yaml)
    except Exception as exc:  # noqa: BLE001 — bad/unparseable JD profile
        return FitOutcome(status="invalid", fit=None, fit_run_id=None,
                          message=f"Could not load the JD profile: {exc}",
                          telemetry=[])

    try:
        fit_result = score_candidate(candidate, jd_profile, jd_method, backend,
                                     ref_date=ref_date or jd_method.ref_date)
    except RateLimitError as exc:
        return FitOutcome(status="rate_limited", fit=None, fit_run_id=None,
                          message=str(exc), telemetry=[])
    except TransientBackendError as exc:
        return FitOutcome(status="transient", fit=None, fit_run_id=None,
                          message=str(exc), telemetry=[])

    # persist: resume + candidate_record + profile (JD) + fit_run
    name = resume_name or candidate.get("profile", {}).get("anonymized_name") or "Candidate"
    resume_id = _new_id("resume")
    store.save_resume(ResumeRow(
        resume_id=resume_id, workspace_id=workspace_id, name=name,
        raw_text=resume_text, candidate_json=candidate, created_at=_now_iso()))

    record_id = _new_id("rec")
    store.save_candidate_record(CandidateRecordRow(
        record_id=record_id, workspace_id=workspace_id, resume_id=resume_id,
        candidate_json=candidate, created_at=_now_iso()))

    profile_id = _new_id("profile")
    store.save_profile(ProfileRow(
        profile_id=profile_id, workspace_id=workspace_id, name=jd_label,
        profile_yaml=jd_profile_yaml, created_at=_now_iso()))

    fit_run_id = _new_id("run")
    store.save_fit_run(FitRunRow(
        run_id=fit_run_id, workspace_id=workspace_id, profile_id=profile_id,
        record_id=record_id, result_json=fit_result.to_dict(),
        overall=fit_result.overall, created_at=_now_iso()))

    return FitOutcome(status="ok", fit=fit_result, fit_run_id=fit_run_id,
                      message="", telemetry=[])
