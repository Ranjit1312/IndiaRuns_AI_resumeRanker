"""Contract test for the Validator seam (Candidate C).

Both adapters — `EngineProfileValidator` (harness/validate.py) and
`JsonSchemaValidator` (harness/candidate_fields.py) — implement the same
`(dict) -> ValidationResult` seam. This test proves they satisfy the
`Validator` Protocol and behave identically on the contract: `ok` on a
known-good dict, and the correct `field`/`top` on a known-bad dict.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from harness.candidate_fields import JsonSchemaValidator, validate_candidate
from harness.validate import EngineProfileValidator, Validator, validate_profile_dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
JD_PATH = os.path.join(_ROOT, "data", "eval_jds", "stripe_backend-software-engineer",
                       "jd_profile.yaml")


def _good_profile() -> dict:
    with open(JD_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _good_candidate() -> dict:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "XYZ Candidate", "headline": "Backend Engineer",
            "summary": "Backend engineer.", "location": "Bengaluru", "country": "India",
            "years_of_experience": 5, "current_title": "Senior Backend Engineer",
            "current_company": "PayCo", "current_company_size": "1001-5000",
            "current_industry": "Fintech",
        },
        "career_history": [
            {"company": "PayCo", "title": "Senior Backend Engineer",
             "start_date": "2023-01-01", "end_date": None, "duration_months": 30,
             "is_current": True, "industry": "Fintech", "company_size": "1001-5000",
             "description": "Built APIs."},
        ],
        "education": [
            {"institution": "IIT Bombay", "degree": "B.Tech",
             "field_of_study": "Computer Science", "start_year": 2015, "end_year": 2019,
             "tier": "unknown"},
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 0},
        ],
        "redrob_signals": {
            "profile_completeness_score": 50, "signup_date": "2025-01-01",
            "last_active_date": "2026-01-01", "open_to_work_flag": True,
            "profile_views_received_30d": 0, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 0,
            "skill_assessment_scores": {}, "connection_count": 0,
            "endorsements_received": 0, "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
            "preferred_work_mode": "flexible", "willing_to_relocate": True,
            "github_activity_score": -1, "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1, "verified_email": False,
            "verified_phone": False, "linkedin_connected": False,
        },
    }


def test_adapters_satisfy_validator_protocol():
    assert isinstance(EngineProfileValidator(), Validator)
    assert isinstance(JsonSchemaValidator(), Validator)


def test_engine_profile_validator_ok_on_good_dict():
    v = EngineProfileValidator()
    res = v.validate(_good_profile())
    assert res.ok, res.error
    # back-compat wrapper agrees
    assert validate_profile_dict(_good_profile()).ok


def test_engine_profile_validator_reports_field_and_top_on_bad_dict():
    prof = _good_profile()
    prof["role"]["title"] = ""   # schema requires non-empty title (minLength)
    v = EngineProfileValidator()
    res = v.validate(prof)
    assert not res.ok
    assert res.top == "role"
    assert res.field and res.field.startswith("role")


def test_json_schema_validator_ok_on_good_dict():
    v = JsonSchemaValidator()
    res = v.validate(_good_candidate())
    assert res.ok, res.error
    # back-compat wrapper agrees
    assert validate_candidate(_good_candidate()).ok


def test_json_schema_validator_reports_field_and_top_on_bad_dict():
    cand = _good_candidate()
    cand["career_history"][0]["duration_months"] = "not-a-number"   # wrong type
    v = JsonSchemaValidator()
    res = v.validate(cand)
    assert not res.ok
    assert res.top == "career_history"
    assert res.field and res.field.startswith("career_history")


if __name__ == "__main__":
    for fn in [test_adapters_satisfy_validator_protocol,
               test_engine_profile_validator_ok_on_good_dict,
               test_engine_profile_validator_reports_field_and_top_on_bad_dict,
               test_json_schema_validator_ok_on_good_dict,
               test_json_schema_validator_reports_field_and_top_on_bad_dict]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all ok")
