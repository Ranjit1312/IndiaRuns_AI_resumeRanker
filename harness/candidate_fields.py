"""Field specs, sentinels, and coercion helpers for the resume->candidate seam.

Mirrors `harness/schema_fields.py`'s discipline for the JD side, but for
`candidate_schema.json` (Phase 2, Part C2). Two jobs:

  1. Safe DEFAULTS for the 23 `redrob_signals` a resume can never supply
     (platform activity: response rates, assessment scores, github activity,
     verification flags, ...). The sentinel policy is chosen so these
     defaults keep the fit engine's gates NEUTRAL rather than punitive — see
     `redrob_ranker/fit.py`'s availability/integrity/notice math, which reads
     exactly these fields (docs/PHASE2_SPEC.md "Sentinel policy").
  2. `validate_candidate(cand) -> ValidationResult` against the authoritative
     `candidate_schema.json` (jsonschema Draft-7), mirroring
     `harness/validate.py`'s field-of-failure parsing so the resume RLM root
     can repair/re-sentinel just the failing block.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SCHEMA_PATH = os.path.join(_ROOT, "candidate_schema.json")

COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1000",
                 "1001-5000", "5001-10000", "10001+"]
WORK_MODES = ["remote", "hybrid", "onsite", "flexible"]
PROFICIENCIES = ["beginner", "intermediate", "advanced", "expert"]

_GITHUB_RE = re.compile(r"github\.com/[A-Za-z0-9_-]+", re.I)
_YEARS_LEFT_RE = re.compile(
    r"(\d{1,3})\s*(?:day|days)\s*(?:notice|left|remaining)", re.I)
_MONTH_NOTICE_RE = re.compile(
    r"(\d{1,2})\s*(?:month|months)\s*notice", re.I)


# ---------------------------------------------------------------------------
# candidate_id
# ---------------------------------------------------------------------------
def new_candidate_id(rng: "random.Random | None" = None) -> str:
    """A schema-legal candidate_id: `CAND_` + 7 digits."""
    r = rng or random
    return f"CAND_{r.randint(0, 9_999_999):07d}"


# ---------------------------------------------------------------------------
# redrob_signals sentinels — a resume cannot supply platform-activity data.
# Policy (see module docstring): every default keeps the engine's gates
# NEUTRAL (not punitive) unless the resume text itself provides evidence.
# ---------------------------------------------------------------------------
def default_redrob_signals(ref_date: "str | None" = None) -> dict:
    """Neutral sentinel values for all 23 redrob_signals.

    ref_date anchors last_active_date (avoids the dormancy/availability damp
    a resume-only candidate would otherwise trip — a resume has no platform
    activity to report, so we treat "just compiled" as "active today").
    """
    today = _parse_date(ref_date) or date.today()
    signup = today - timedelta(days=365)
    return {
        "profile_completeness_score": 50,
        "signup_date": signup.isoformat(),
        "last_active_date": today.isoformat(),
        "open_to_work_flag": True,
        "profile_views_received_30d": 0,
        "applications_submitted_30d": 0,
        # candidate_schema.json requires 0<=recruiter_response_rate<=1 (unlike
        # github_activity_score/offer_acceptance_rate, this field has NO -1
        # "missing" sentinel in the schema). 0.5 is the schema-valid neutral
        # midpoint: it sits safely above method.availability.dormancy.rrr_below
        # (0.2) so a resume-only candidate never trips the dormancy damp, and
        # it does not inflate the availability score the way rrr=1.0 would.
        "recruiter_response_rate": 0.5,
        "avg_response_time_hours": 0,
        "skill_assessment_scores": {},   # -> assess_strength=0 -> m_assess=1.0 (neutral)
        "connection_count": 0,
        "endorsements_received": 0,
        "notice_period_days": 60,
        "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
        "preferred_work_mode": "flexible",
        "willing_to_relocate": True,
        "github_activity_score": -1,     # -1 unless a GitHub URL is parsed
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": 0.5,
        "offer_acceptance_rate": -1,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
    }


def coerce_redrob_signals(parsed: "dict | None", resume_text: str,
                          ref_date: "str | None" = None) -> dict:
    """Merge whatever a leaf extracted (rare — most signals aren't in a resume)
    over the neutral sentinel defaults, then fold in resume-derivable hints
    (GitHub URL -> github_activity_score placeholder presence; notice period
    phrasing -> notice_period_days)."""
    out = default_redrob_signals(ref_date)
    d = parsed if isinstance(parsed, dict) else {}

    if isinstance(d.get("notice_period_days"), (int, float)):
        out["notice_period_days"] = _clamp_int(d["notice_period_days"], 0, 180, 60)
    else:
        parsed_notice = _parse_notice_days(resume_text)
        if parsed_notice is not None:
            out["notice_period_days"] = parsed_notice

    if isinstance(d.get("preferred_work_mode"), str) and d["preferred_work_mode"] in WORK_MODES:
        out["preferred_work_mode"] = d["preferred_work_mode"]
    if isinstance(d.get("willing_to_relocate"), bool):
        out["willing_to_relocate"] = d["willing_to_relocate"]
    if isinstance(d.get("expected_salary_range_inr_lpa"), dict):
        sal = d["expected_salary_range_inr_lpa"]
        lo = _as_number(sal.get("min"), 0.0)
        hi = _as_number(sal.get("max"), 0.0)
        if lo < 0:
            lo = 0.0
        if hi < lo:
            hi = lo
        out["expected_salary_range_inr_lpa"] = {"min": lo, "max": hi}

    if _GITHUB_RE.search(resume_text or ""):
        # A GitHub profile was found in the resume text -> no longer "unknown";
        # use a neutral mid score rather than fabricating platform-derived
        # commit/PR/star activity we cannot actually observe.
        out["github_activity_score"] = 40.0

    for flag in ("verified_email", "verified_phone", "linkedin_connected"):
        if isinstance(d.get(flag), bool):
            out[flag] = d[flag]

    return out


def _parse_notice_days(text: str) -> "int | None":
    if not text:
        return None
    m = _YEARS_LEFT_RE.search(text)
    if m:
        return _clamp_int(int(m.group(1)), 0, 180, 60)
    m = _MONTH_NOTICE_RE.search(text)
    if m:
        return _clamp_int(int(m.group(1)) * 30, 0, 180, 60)
    if re.search(r"immediate(?:ly)?\s*(?:joiner|availab)", text, re.I):
        return 0
    return None


# ---------------------------------------------------------------------------
# career_history coercion — compute duration/is_current, default size/industry
# ---------------------------------------------------------------------------
def coerce_career_entry(raw: dict, *, ref_date: "str | None" = None) -> dict:
    """Fill in a single career_history entry so it always satisfies the
    schema's required keys, computing duration_months from dates when the
    leaf didn't supply one and defaulting company_size/industry."""
    raw = raw if isinstance(raw, dict) else {}
    company = str(raw.get("company") or "Unknown Company").strip() or "Unknown Company"
    title = str(raw.get("title") or "Unknown Title").strip() or "Unknown Title"
    start_date = _coerce_date_str(raw.get("start_date")) or _default_start_date(ref_date)
    end_raw = raw.get("end_date")
    is_current = bool(raw.get("is_current")) or end_raw in (None, "", "present", "Present",
                                                            "current", "Current")
    end_date = None if is_current else _coerce_date_str(end_raw)
    if not is_current and end_date is None:
        # couldn't parse an end date and it wasn't flagged current -> treat as
        # ongoing rather than inventing a fake end date.
        is_current = True

    dur = raw.get("duration_months")
    if not isinstance(dur, (int, float)) or dur < 0:
        dur = _duration_months(start_date, end_date, ref_date if is_current else None)
    dur = max(0, int(round(dur)))

    size = raw.get("company_size")
    if size not in COMPANY_SIZES:
        size = "201-500"
    industry = str(raw.get("industry") or "Technology").strip() or "Technology"
    description = str(raw.get("description") or "").strip()
    if not description:
        description = f"{title} at {company}."

    return {
        "company": company,
        "title": title,
        "start_date": start_date,
        "end_date": end_date,
        "duration_months": dur,
        "is_current": is_current,
        "industry": industry,
        "company_size": size,
        "description": description,
    }


def coerce_career_history(raw_list, *, ref_date: "str | None" = None) -> list:
    entries = [coerce_career_entry(r, ref_date=ref_date)
               for r in (raw_list or []) if isinstance(r, dict)]
    if not entries:
        entries = [coerce_career_entry({}, ref_date=ref_date)]
    return entries[:10]


# ---------------------------------------------------------------------------
# education / skills coercion
# ---------------------------------------------------------------------------
def coerce_education_entry(raw: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    institution = str(raw.get("institution") or "Unknown Institution").strip() \
        or "Unknown Institution"
    degree = str(raw.get("degree") or "Bachelor's").strip() or "Bachelor's"
    field_of_study = str(raw.get("field_of_study") or "General Studies").strip() \
        or "General Studies"
    start_year = _clamp_int(raw.get("start_year"), 1970, 2030, 2015)
    end_year = _clamp_int(raw.get("end_year"), 1970, 2035, max(start_year, 2019))
    if end_year < start_year:
        end_year = start_year
    out = {
        "institution": institution, "degree": degree,
        "field_of_study": field_of_study,
        "start_year": start_year, "end_year": end_year,
    }
    grade = raw.get("grade")
    if grade is not None:
        out["grade"] = str(grade)
    tier = raw.get("tier")
    out["tier"] = tier if tier in ("tier_1", "tier_2", "tier_3", "tier_4", "unknown") \
        else "unknown"
    return out


def coerce_education(raw_list) -> list:
    return [coerce_education_entry(r) for r in (raw_list or [])
            if isinstance(r, dict)][:5]


def coerce_skill_entry(raw: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    name = str(raw.get("name") or "").strip()
    if not name:
        return {}
    prof = raw.get("proficiency")
    if prof not in PROFICIENCIES:
        prof = "intermediate"
    endorsements = _clamp_int(raw.get("endorsements"), 0, 10_000, 0)
    out = {"name": name, "proficiency": prof, "endorsements": endorsements}
    dur = raw.get("duration_months")
    if isinstance(dur, (int, float)) and dur >= 0:
        out["duration_months"] = int(round(dur))
    return out


def coerce_skills(raw_list) -> list:
    out = []
    seen = set()
    for r in (raw_list or []):
        s = coerce_skill_entry(r if isinstance(r, dict) else {})
        if s and s["name"].lower() not in seen:
            seen.add(s["name"].lower())
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# profile block
# ---------------------------------------------------------------------------
def coerce_profile(raw: dict) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    yoe = _as_number(raw.get("years_of_experience"), 0.0)
    yoe = max(0.0, min(50.0, yoe))
    size = raw.get("current_company_size")
    if size not in COMPANY_SIZES:
        size = "201-500"
    return {
        "anonymized_name": str(raw.get("anonymized_name") or "XYZ Candidate").strip()
            or "XYZ Candidate",
        "headline": str(raw.get("headline") or "Professional").strip() or "Professional",
        "summary": str(raw.get("summary") or "").strip(),
        "location": str(raw.get("location") or "Unknown").strip() or "Unknown",
        "country": str(raw.get("country") or "India").strip() or "India",
        "years_of_experience": yoe,
        "current_title": str(raw.get("current_title") or "Professional").strip()
            or "Professional",
        "current_company": str(raw.get("current_company") or "Unknown Company").strip()
            or "Unknown Company",
        "current_company_size": size,
        "current_industry": str(raw.get("current_industry") or "Technology").strip()
            or "Technology",
    }


def append_to_summary(profile: dict, extra_text: str) -> dict:
    """Append project text to profile.summary WITHOUT touching career_history
    (the faithfulness rule: projects must not distort tenure/hopper math)."""
    extra_text = (extra_text or "").strip()
    if not extra_text:
        return profile
    base = (profile.get("summary") or "").strip()
    profile = dict(profile)
    profile["summary"] = (base + ("\n\n" if base else "") + extra_text).strip()
    return profile


# ---------------------------------------------------------------------------
# small numeric / date helpers
# ---------------------------------------------------------------------------
def _as_number(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _parse_date(d) -> "date | None":
    if not d:
        return None
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _coerce_date_str(d) -> "str | None":
    parsed = _parse_date(d)
    return parsed.isoformat() if parsed else None


def _default_start_date(ref_date: "str | None") -> str:
    ref = _parse_date(ref_date) or date.today()
    return (ref - timedelta(days=365)).isoformat()


def _duration_months(start: "str | None", end: "str | None",
                      current_ref: "str | None") -> float:
    s = _parse_date(start)
    e = _parse_date(end) if end else (_parse_date(current_ref) if current_ref else None)
    if s is None or e is None:
        return 12.0
    months = (e.year - s.year) * 12 + (e.month - s.month) + (e.day - s.day) / 30.0
    return max(0.0, months)


# ---------------------------------------------------------------------------
# validation — authoritative, against candidate_schema.json (jsonschema)
# ---------------------------------------------------------------------------
_FIELD_RE = re.compile(r"^(\S+)")


@dataclass
class ValidationResult:
    ok: bool
    error: "str | None" = None
    field: "str | None" = None     # offending dotted path, if parseable
    top: "str | None" = None       # top-level block (e.g. "redrob_signals", "career_history")


def _load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class JsonSchemaValidator:
    """Adapter: validates a dict against a JSON-Schema file (Draft-7).

    Mirrors `harness/validate.py`'s `EngineProfileValidator`: same
    `(dict) -> ValidationResult` seam, different mechanism (`jsonschema`
    instead of the engine's own loader). Defaults to `candidate_schema.json`
    but works for any Draft-7 schema file.
    """

    def __init__(self, schema_path: str = SCHEMA_PATH) -> None:
        self.schema_path = schema_path

    def validate(self, d: dict) -> ValidationResult:
        try:
            import jsonschema
        except ImportError as exc:  # pragma: no cover - jsonschema is a core dep
            return ValidationResult(ok=False, error=f"jsonschema not importable: {exc}")

        try:
            schema = _load_schema() if self.schema_path == SCHEMA_PATH else \
                json.load(open(self.schema_path, "r", encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ValidationResult(ok=False, error=f"could not load schema: {exc}")

        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(d), key=lambda e: list(e.path))
        if not errors:
            return ValidationResult(ok=True)
        e = errors[0]
        loc = ".".join(str(p) for p in e.path) or "<root>"
        top = str(e.path[0]) if e.path else (loc if loc != "<root>" else None)
        return ValidationResult(ok=False, error=e.message, field=loc, top=top)


def validate_candidate(cand: dict, schema_path: str = SCHEMA_PATH) -> ValidationResult:
    """Validate a candidate dict against candidate_schema.json (Draft-7).

    Mirrors `harness/validate.py`'s field-of-failure parsing: returns the
    dotted path of the first schema violation (jsonschema's own ordering) and
    its top-level block name, so a caller can repair/re-sentinel just that
    block without re-deriving the whole candidate.

    Thin back-compat wrapper — delegates to `JsonSchemaValidator`.
    """
    return JsonSchemaValidator(schema_path).validate(cand)
