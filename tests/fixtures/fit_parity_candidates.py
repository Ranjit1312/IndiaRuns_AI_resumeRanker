"""Fixture candidates for the fit-scorer parity pin (Candidate D).

Three hand-built, fully deterministic candidate dicts spanning strong /
borderline / weak-with-gate-trips, scored against the gold Stripe
backend-engineer JD profile (data/eval_jds/stripe_backend-software-engineer/
jd_profile.yaml + jd/method_config.yaml). Shared by:

  - scripts/gen_fit_parity_golden.py (regenerates the golden JSON)
  - tests/test_fit_parity.py (asserts compute_components reproduces it)

None of these candidates touch embeddings/BM25/mm() — that's the point:
`compute_components` needs none of that, so the values here are exactly
reproducible byte-for-byte from `candidate + profile + method` alone.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

JD_PATH = os.path.join(_ROOT, "data", "eval_jds", "stripe_backend-software-engineer",
                       "jd_profile.yaml")
METHOD_PATH = os.path.join(_ROOT, "jd", "method_config.yaml")
REF_DATE = "2026-06-06"   # matches jd/method_config.yaml runtime.ref_date


def _career(title, company, start, end, months, description, industry="Software",
           is_current=False, size="201-500"):
    return {
        "company": company, "title": title, "start_date": start, "end_date": end,
        "duration_months": months, "is_current": is_current, "industry": industry,
        "company_size": size, "description": description,
    }


def strong_candidate():
    """Deep, recent, on-domain backend engineer; available; short notice.

    No gates trip: high integrity, high availability, short notice, good
    location fit, tenured (not a hopper), not consulting-only, recent IC role.
    """
    return {
        "candidate_id": "FIX_STRONG_0001",
        "profile": {
            "anonymized_name": "Strong Candidate", "headline": "Backend Engineer, 5 years",
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


def borderline_candidate():
    """Mixed signal: on-domain but junior-ish, average tenure, middling
    availability, mid-length notice, acceptable (not preferred) location.

    No hard gate trips, but no bonuses either — a "middle of the pack" fixture
    distinct from both the strong and weak extremes.
    """
    return {
        "candidate_id": "FIX_BORDERLINE_0002",
        "profile": {
            "anonymized_name": "Borderline Candidate", "headline": "Software Engineer, 2 years",
            "summary": "Software engineer working on internal tools and some backend "
                       "API work for a mid-size product company.",
            "location": "Dublin", "country": "Ireland", "years_of_experience": 2,
            "current_title": "Software Engineer", "current_company": "MidCo",
            "current_company_size": "51-200", "current_industry": "Software",
        },
        "career_history": [
            _career(
                "Software Engineer", "MidCo", "2024-01-01", None, 18,
                "Built internal tooling and some backend API endpoints for "
                "reporting. Occasional on-call shadowing.",
                is_current=True, industry="Software",
            ),
            _career(
                "Junior Developer", "SmallCo", "2022-07-01", "2023-12-01", 17,
                "Worked on a small internal dashboard app, mostly front-end with "
                "a bit of API glue code.",
                industry="Software",
            ),
        ],
        "education": [
            {"institution": "Trinity College Dublin", "degree": "B.Sc",
             "field_of_study": "Computer Science", "start_year": 2018, "end_year": 2022},
        ],
        "skills": [
            {"name": "Python", "proficiency": "intermediate", "endorsements": 2, "duration_months": 24},
            {"name": "Docker", "proficiency": "beginner", "endorsements": 1, "duration_months": 6},
        ],
        "redrob_signals": {
            "profile_completeness_score": 60, "signup_date": "2023-01-01",
            "last_active_date": "2026-04-01", "open_to_work_flag": True,
            "profile_views_received_30d": 5, "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.4, "avg_response_time_hours": 30,
            "skill_assessment_scores": {}, "connection_count": 80,
            "endorsements_received": 3,
            "notice_period_days": 45,
            "expected_salary_range_inr_lpa": {"min": 15, "max": 20},
            "preferred_work_mode": "hybrid", "willing_to_relocate": False,
            "github_activity_score": 20, "search_appearance_30d": 2,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": False, "linkedin_connected": True,
        },
    }


def weak_candidate():
    """Consulting-only, job-hopper, stale IC role, long notice, dormant/low RR,
    out-of-domain career text — trips only_consulting, job_hopper AND
    stale_ic_role (all enabled for this JD) plus notice/availability gates."""
    return {
        "candidate_id": "FIX_WEAK_0003",
        "profile": {
            "anonymized_name": "Weak Candidate", "headline": "Consultant",
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


# name -> builder, in a fixed order so golden generation/loading is stable.
FIXTURES = {
    "strong": strong_candidate,
    "borderline": borderline_candidate,
    "weak": weak_candidate,
}
