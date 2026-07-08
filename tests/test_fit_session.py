"""Offline tests for fit_session.parse_resume / run_fit (MockBackend + a
stub embedding backend for score_candidate, MemoryStore via get_store() —
no API key, no network).

Mirrors tests/test_resume.py's MockBackend responder pattern for
`parse_resume`, tests/test_fit.py's MockEmbedBackend pattern for the
embedding side of `run_fit`, and tests/test_store.py's get_store() /
Workspace usage for persistence assertions.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.backends import MockBackend, RateLimitError
from fit_session import FitOutcome, ParsedResume, parse_resume, run_fit
from store import get_store
from store.schema import Workspace as WorkspaceRow

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
JD_PATH = os.path.join(_ROOT, "data", "eval_jds", "stripe_backend-software-engineer",
                       "jd_profile.yaml")
REF_DATE = "2026-06-06"   # matches jd/method_config.yaml runtime.ref_date
WS = "ws_test_fit_session"

RESUME = """XYZ Candidate
abc@example.com | +91-XXXXXXXXXX | Bengaluru, India
github.com/xyzcandidate

SUMMARY
Backend engineer with 5 years of experience building APIs and payment systems.

EXPERIENCE
Senior Backend Engineer, PayCo (Jan 2023 - Present)
- Designed and built large-scale APIs handling billions of payment requests.
- Led on-call rotation and mentored 3 engineers.

Backend Engineer, Shopz (Jun 2020 - Dec 2022)
- Built distributed microservices and internal dashboards.

EDUCATION
IIT Bombay, B.Tech Computer Science, 2015-2019

SKILLS
Python, Kubernetes, gRPC, AWS, Docker
"""


def _jd_profile_yaml() -> str:
    with open(JD_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# generate() responder — mirrors tests/test_resume.py's _good
# --------------------------------------------------------------------------- #
def _good_generate(prompt, system):
    if '"headline"' in prompt:
        return json.dumps({
            "headline": "Senior Backend Engineer", "summary": "Backend engineer.",
            "location": "Bengaluru", "country": "India", "years_of_experience": 5,
            "current_title": "Senior Backend Engineer", "current_company": "PayCo",
            "current_company_size": "1001-5000", "current_industry": "Fintech",
        })
    if "employment/work-experience" in prompt:
        return json.dumps([
            {"company": "PayCo", "title": "Senior Backend Engineer",
             "start_date": "2023-01-01", "end_date": None, "duration_months": 30,
             "is_current": True, "industry": "Fintech", "company_size": "1001-5000",
             "description": "Designed and built large-scale APIs and backend "
                            "services for payment systems, on-call, mentoring."},
            {"company": "Shopz", "title": "Backend Engineer",
             "start_date": "2020-06-01", "end_date": "2022-12-01", "duration_months": 30,
             "is_current": False, "industry": "E-commerce", "company_size": "201-500",
             "description": "Built distributed microservices."},
        ])
    if "education entry" in prompt:
        return json.dumps([
            {"institution": "IIT Bombay", "degree": "B.Tech",
             "field_of_study": "Computer Science", "start_year": 2015, "end_year": 2019,
             "grade": None},
        ])
    if '"proficiency"' in prompt:
        return json.dumps([
            {"name": "Python", "proficiency": "expert", "endorsements": 0,
             "duration_months": 60},
            {"name": "Kubernetes", "proficiency": "advanced", "endorsements": 0,
             "duration_months": 36},
        ])
    if "projects_summary" in prompt:
        return json.dumps({"projects_summary": ""})
    return "{}"


def _rate_limited_generate(prompt, system):
    raise RateLimitError("429 RESOURCE_EXHAUSTED (simulated)")


# --------------------------------------------------------------------------- #
# stub embed() — a Backend that both generate()s (for parse_resume, unused in
# run_fit tests) and embed()s (for score_candidate); mirrors test_fit.py's
# MockEmbedBackend but bundled as one backend so run_fit's single `backend`
# arg can do both jobs where a test needs it.
# --------------------------------------------------------------------------- #
class StubBackend:
    name = "stub"
    supports_response_schema = False

    def __init__(self, generate_fn=_good_generate, dim: int = 64,
                 raise_on_embed: "Exception | None" = None):
        self._generate_fn = generate_fn
        self.dim = dim
        self._raise_on_embed = raise_on_embed

    def generate(self, prompt, system=None, temperature=0.2, max_tokens=2048):
        return self._generate_fn(prompt, system)

    def _vec(self, text: str):
        import numpy as np
        toks = [t for t in text.lower().split() if t.isalpha()]
        v = np.zeros(self.dim)
        for t in toks:
            h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v += rng.normal(size=self.dim)
        if not toks:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v = rng.normal(size=self.dim)
        n = float((v ** 2).sum()) ** 0.5
        return (v / n).tolist() if n else v.tolist()

    def embed(self, texts):
        if self._raise_on_embed is not None:
            raise self._raise_on_embed
        return [self._vec(t) for t in texts]


def _store():
    store = get_store()   # default: MemoryStore, no config/env needed
    store.save_workspace(WorkspaceRow(workspace_id=WS, name="Test WS", created_at="t0"))
    return store


def _career(title, company, start, end, months, description, industry="Software",
           is_current=False, size="201-500"):
    return {
        "company": company, "title": title, "start_date": start, "end_date": end,
        "duration_months": months, "is_current": is_current, "industry": industry,
        "company_size": size, "description": description,
    }


def _strong_candidate() -> dict:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "XYZ Candidate", "headline": "Backend Engineer, 5 years",
            "summary": "Backend software engineer building large-scale APIs and "
                       "payment systems, mentoring engineers, on-call for production.",
            "location": "Bengaluru", "country": "India", "years_of_experience": 5,
            "current_title": "Senior Backend Engineer", "current_company": "PayCo",
            "current_company_size": "1001-5000", "current_industry": "Fintech",
        },
        "career_history": [
            _career(
                "Senior Backend Engineer", "PayCo", "2023-01-01", None, 30,
                "Designed and built large-scale APIs and backend services handling "
                "billions of payment transaction requests. Owned the ledger and "
                "money movement pipeline from scratch. Led on-call rotation, "
                "debugged critical production incidents and wrote postmortems. "
                "Mentored 3 engineers and drove cross-functional collaboration with "
                "product. Used gRPC, Docker, Kubernetes, AWS.",
                is_current=True, industry="Fintech",
            ),
            _career(
                "Backend Engineer", "Shopz", "2020-06-01", "2022-12-01", 30,
                "Built distributed microservices and internal dashboards, designed "
                "APIs, improved engineering standards and tooling for the team.",
                industry="E-commerce",
            ),
        ],
        "education": [
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2015, "end_year": 2019},
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 60},
            {"name": "Kubernetes", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
            {"name": "gRPC", "proficiency": "advanced", "endorsements": 3, "duration_months": 24},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90, "signup_date": "2022-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 20, "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.9, "avg_response_time_hours": 4,
            "skill_assessment_scores": {"backend api design": 85, "system design": 80},
            "connection_count": 300, "endorsements_received": 40,
            "notice_period_days": 15,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 60, "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.8, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }


def _weak_candidate() -> dict:
    return {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "ABC Candidate", "headline": "Consultant",
            "summary": "Graphic design and marketing data entry consultant.",
            "location": "Remote", "country": "Unknown", "years_of_experience": 6,
            "current_title": "Manager", "current_company": "TCS",
            "current_company_size": "10001+", "current_industry": "Consulting",
        },
        "career_history": [
            _career("Manager", "TCS", "2025-01-01", "2025-06-01", 5,
                   "Managed client engagements in graphic design and marketing "
                   "data entry projects for a consulting client.",
                   industry="Consulting"),
            _career("Manager", "Infosys", "2024-06-01", "2024-12-01", 6,
                   "Led sales and marketing data entry consulting engagements.",
                   industry="Consulting"),
            _career("Manager", "Wipro", "2023-11-01", "2024-05-01", 6,
                   "Consulting engagement in graphic design for marketing client.",
                   industry="Consulting"),
            _career("Manager", "Accenture", "2023-04-01", "2023-10-01", 6,
                   "Sales consulting, data entry for marketing client.",
                   industry="Consulting"),
            _career("Manager", "Deloitte", "2022-10-01", "2023-03-01", 5,
                   "Graphic design consulting for a marketing client.",
                   industry="Consulting"),
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 20, "signup_date": "2020-01-01",
            "last_active_date": "2025-01-01", "open_to_work_flag": False,
            "profile_views_received_30d": 0, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.05, "avg_response_time_hours": 200,
            "skill_assessment_scores": {}, "connection_count": 5,
            "endorsements_received": 0, "notice_period_days": 150,
            "expected_salary_range_inr_lpa": {"min": 10, "max": 12},
            "preferred_work_mode": "remote", "willing_to_relocate": False,
            "github_activity_score": -1, "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.1,
            "offer_acceptance_rate": -1, "verified_email": False,
            "verified_phone": False, "linkedin_connected": False,
        },
    }


# --------------------------------------------------------------------------- #
# parse_resume
# --------------------------------------------------------------------------- #
def test_parse_resume_returns_schema_valid_candidate_with_telemetry():
    backend = StubBackend(_good_generate)
    parsed = parse_resume(RESUME, backend, ref_date=REF_DATE)

    assert isinstance(parsed, ParsedResume)
    assert parsed.validation.ok, parsed.validation.error
    assert isinstance(parsed.candidate, dict)
    assert parsed.candidate.get("candidate_id", "").startswith("CAND_")
    assert parsed.candidate["career_history"], "expected parsed career history"
    assert isinstance(parsed.health, dict)
    assert parsed.telemetry, "expected populated telemetry (HarnessLogger entries)"
    for entry in parsed.telemetry:
        assert "leaf" in entry and "ok" in entry


def test_parse_resume_fires_on_event():
    backend = StubBackend(_good_generate)
    events = []
    parse_resume(RESUME, backend, on_event=lambda e: events.append(e), ref_date=REF_DATE)
    assert events, "expected on_event to fire for each leaf call"


# --------------------------------------------------------------------------- #
# run_fit — success path
# --------------------------------------------------------------------------- #
def test_run_fit_ok_strong_candidate_persists_fit_run():
    store = _store()
    backend = StubBackend()
    outcome = run_fit(_strong_candidate(), _jd_profile_yaml(), backend, store,
                      workspace_id=WS, ref_date=REF_DATE, jd_label="stripe-backend")

    assert isinstance(outcome, FitOutcome)
    assert outcome.status == "ok", outcome.message
    assert outcome.fit is not None
    assert 0.0 <= outcome.fit.overall <= 100.0
    assert outcome.fit_run_id

    runs = store.list_fit_runs(WS)
    assert len(runs) == 1
    assert runs[0].run_id == outcome.fit_run_id
    assert runs[0].overall == outcome.fit.overall
    # persistence unification: resume + candidate_record + fit_run all written
    assert len(store.list_resumes(WS)) == 1
    assert len(store.list_candidate_records(WS)) == 1


def test_run_fit_weak_scores_lower_than_strong():
    store = _store()
    backend = StubBackend()
    strong = run_fit(_strong_candidate(), _jd_profile_yaml(), backend, store,
                     workspace_id=WS, ref_date=REF_DATE)
    weak = run_fit(_weak_candidate(), _jd_profile_yaml(), backend, store,
                   workspace_id=WS, ref_date=REF_DATE)

    assert strong.status == "ok" and weak.status == "ok"
    assert strong.fit.overall > weak.fit.overall


# --------------------------------------------------------------------------- #
# run_fit — invalid candidate
# --------------------------------------------------------------------------- #
def test_run_fit_invalid_candidate_persists_nothing():
    store = _store()
    backend = StubBackend()
    bad_candidate = {"candidate_id": "not-even-close-to-schema-valid"}

    outcome = run_fit(bad_candidate, _jd_profile_yaml(), backend, store,
                      workspace_id=WS, ref_date=REF_DATE)

    assert outcome.status == "invalid"
    assert outcome.fit is None
    assert outcome.fit_run_id is None
    assert outcome.message
    assert store.list_fit_runs(WS) == []
    assert store.list_resumes(WS) == []
    assert store.list_candidate_records(WS) == []


# --------------------------------------------------------------------------- #
# run_fit — rate limited
# --------------------------------------------------------------------------- #
def test_run_fit_rate_limited_persists_nothing():
    store = _store()
    backend = StubBackend(raise_on_embed=RateLimitError("429 RESOURCE_EXHAUSTED (simulated)"))

    outcome = run_fit(_strong_candidate(), _jd_profile_yaml(), backend, store,
                      workspace_id=WS, ref_date=REF_DATE)

    assert outcome.status == "rate_limited"
    assert outcome.fit is None
    assert outcome.fit_run_id is None
    assert "429" in outcome.message or "RESOURCE_EXHAUSTED" in outcome.message
    assert store.list_fit_runs(WS) == []
    assert store.list_resumes(WS) == []
    assert store.list_candidate_records(WS) == []
