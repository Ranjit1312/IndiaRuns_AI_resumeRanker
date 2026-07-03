"""Offline tests for redrob_ranker.fit.score_candidate (MockEmbedBackend — no
API key, no network).

Loads the gold Stripe backend-engineer profile/method
(data/eval_jds/stripe_backend-software-engineer/jd_profile.yaml +
jd/method_config.yaml) and scores a hand-built STRONG candidate (deep,
recent, on-domain, tenured, available, short notice) against a WEAK candidate
engineered to trip several gates (job-hopper, stale IC role, long notice,
dormant/low response, out-of-domain/consulting-only history). Proves the
FitResult structure and that strong > weak, and that damping gates / red
flags surface in `gaps`.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redrob_ranker as rr
from redrob_ranker.fit import FitResult, score_candidate

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
JD_PATH = os.path.join(_ROOT, "data", "eval_jds", "stripe_backend-software-engineer",
                       "jd_profile.yaml")
METHOD_PATH = os.path.join(_ROOT, "jd", "method_config.yaml")
REF_DATE = "2026-06-06"   # matches jd/method_config.yaml runtime.ref_date


class MockEmbedBackend:
    """Deterministic offline embedder: hash-seeded unit vectors.

    Text that shares more tokens with a signal query lands closer to that
    query's vector (via a small bag-of-tokens component blended with the
    hash noise), so the strong/weak fixtures produce a real dense-similarity
    spread without any network call.
    """
    name = "mock-embed"
    supports_response_schema = False

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = []

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
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]


def _load_profile():
    return rr.load(JD_PATH, METHOD_PATH)


def _career(title, company, start, end, months, description, industry="Software",
           is_current=False, size="201-500"):
    return {
        "company": company, "title": title, "start_date": start, "end_date": end,
        "duration_months": months, "is_current": is_current, "industry": industry,
        "company_size": size, "description": description,
    }


def _strong_candidate():
    """Deep, recent, on-domain backend engineer; available; short notice."""
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


def _weak_candidate():
    """Consulting-only, job-hopper, stale IC role, long notice, dormant/low RR,
    out-of-domain career text — engineered to trip multiple gates + red flags."""
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


def test_fit_result_structure():
    profile, method = _load_profile()
    backend = MockEmbedBackend()
    res = score_candidate(_strong_candidate(), profile, method, backend, ref_date=REF_DATE)

    assert isinstance(res, FitResult)
    assert 0.0 <= res.overall <= 100.0
    assert res.per_signal and len(res.per_signal) == len(profile.signals)
    for sig in res.per_signal:
        assert sig.id and sig.label
        assert isinstance(sig.dense, float)
        assert isinstance(sig.lexical, float)
        assert isinstance(sig.evidence, float)
        assert isinstance(sig.weight, float)
        assert isinstance(sig.contribution, float)
    gate_names = {g.name for g in res.gates}
    assert gate_names == {"integrity", "availability", "notice", "location"}
    for g in res.gates:
        assert isinstance(g.value, float)
        assert isinstance(g.damped, bool)
    assert isinstance(res.red_flags, list)
    assert isinstance(res.gaps, list)

    d = res.to_dict()
    assert d["overall"] == res.overall
    assert set(d.keys()) == {"overall", "candidate_id", "role_title",
                             "per_signal", "gates", "red_flags", "gaps"}
    assert d["per_signal"][0]["id"] == res.per_signal[0].id


def test_strong_beats_weak():
    profile, method = _load_profile()
    backend = MockEmbedBackend()
    strong = score_candidate(_strong_candidate(), profile, method, backend, ref_date=REF_DATE)
    weak = score_candidate(_weak_candidate(), profile, method, backend, ref_date=REF_DATE)

    assert strong.overall > weak.overall, (
        f"expected strong ({strong.overall}) > weak ({weak.overall})")


def test_weak_candidate_trips_gates_and_gaps():
    profile, method = _load_profile()
    backend = MockEmbedBackend()
    weak = score_candidate(_weak_candidate(), profile, method, backend, ref_date=REF_DATE)

    # only_consulting is enabled for this JD and the weak candidate is 100%
    # consulting-company tenure -> should fire as a red flag.
    assert "only_consulting" in weak.red_flags
    # notice_period_days=150 exceeds every finite tier in method.notice_tiers
    # -> the notice gate should show as damped.
    notice_gate = next(g for g in weak.gates if g.name == "notice")
    assert notice_gate.damped
    # long notice period and the fired red flag should both surface as coaching text.
    gaps_text = " ".join(weak.gaps).lower()
    assert "notice period" in gaps_text
    assert "consulting" in gaps_text


def test_mock_backend_used_no_network():
    profile, method = _load_profile()
    backend = MockEmbedBackend()
    score_candidate(_strong_candidate(), profile, method, backend, ref_date=REF_DATE)
    assert backend.calls, "expected embed() to have been called at least once"


def test_score_candidate_requires_embed_backend():
    profile, method = _load_profile()

    class NoEmbedBackend:
        name = "no-embed"
        supports_response_schema = False

        def generate(self, prompt, system=None, temperature=0.2, max_tokens=2048):
            return "{}"

    try:
        score_candidate(_strong_candidate(), profile, method, NoEmbedBackend(),
                        ref_date=REF_DATE)
        assert False, "expected ValueError for a backend without embed()"
    except ValueError as exc:
        assert "embed" in str(exc)
