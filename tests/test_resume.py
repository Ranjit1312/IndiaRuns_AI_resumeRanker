"""Offline tests for harness.resume.compile_resume (MockBackend — no API key,
no network).

Proves compile_resume ALWAYS returns a schema-valid candidate (good model,
useless model, rate-limited model), that projects land in profile.summary
NOT career_history (the faithfulness rule), that all 23 redrob_signals
sentinels are present, and that telemetry/on_event fire per leaf call.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.backends import MockBackend, RateLimitError
from harness.candidate_fields import validate_candidate
from harness.logging_utils import HarnessLogger
from harness.resume import compile_resume

REF_DATE = "2026-07-03"

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

PROJECTS
Open-source rate limiter (github.com/xyzcandidate/ratelimiter)
- Built a token-bucket rate limiter in Go, 200+ stars.
Personal finance tracker
- Side project using React and a Python Flask API.

EDUCATION
IIT Bombay, B.Tech Computer Science, 2015-2019

SKILLS
Python, Kubernetes, gRPC, AWS, Docker
"""

_ALL_23_SIGNALS = [
    "profile_completeness_score", "signup_date", "last_active_date",
    "open_to_work_flag", "profile_views_received_30d", "applications_submitted_30d",
    "recruiter_response_rate", "avg_response_time_hours", "skill_assessment_scores",
    "connection_count", "endorsements_received", "notice_period_days",
    "expected_salary_range_inr_lpa", "preferred_work_mode", "willing_to_relocate",
    "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d",
    "interview_completion_rate", "offer_acceptance_rate", "verified_email",
    "verified_phone", "linkedin_connected",
]


def _good(prompt, system):
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
             "description": "Designed and built large-scale APIs."},
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
        return json.dumps({
            "projects_summary": "Built an open-source token-bucket rate limiter in "
                                "Go with 200+ stars, and a personal finance tracker "
                                "using React and Flask.",
        })
    return "{}"


def _bad(prompt, system):
    return "Sorry, I can't help with that."   # never valid JSON


def _rate_limited(prompt, system):
    raise RateLimitError("429 RESOURCE_EXHAUSTED (simulated)")


def test_good_model_produces_valid_candidate():
    res = compile_resume(RESUME, MockBackend(_good), ref_date=REF_DATE)
    assert res.validation.ok, res.validation.error
    vr = validate_candidate(res.candidate)
    assert vr.ok, vr.error


def test_candidate_id_format():
    res = compile_resume(RESUME, MockBackend(_good), ref_date=REF_DATE)
    import re
    assert re.match(r"^CAND_[0-9]{7}$", res.candidate["candidate_id"])


def test_projects_land_in_summary_not_career_history():
    res = compile_resume(RESUME, MockBackend(_good), ref_date=REF_DATE)
    cand = res.candidate
    summary = cand["profile"]["summary"].lower()
    assert "rate limiter" in summary or "finance tracker" in summary, \
        "projects text should be folded into profile.summary"
    for job in cand["career_history"]:
        blob = (job["company"] + " " + job["title"] + " " + job["description"]).lower()
        assert "rate limiter" not in blob
        assert "finance tracker" not in blob
    # only the two real employers should appear as career_history entries
    companies = {j["company"] for j in cand["career_history"]}
    assert companies == {"PayCo", "Shopz"}


def test_all_23_redrob_signals_present_with_sentinels():
    res = compile_resume(RESUME, MockBackend(_good), ref_date=REF_DATE)
    sig = res.candidate["redrob_signals"]
    for key in _ALL_23_SIGNALS:
        assert key in sig, f"missing redrob_signal sentinel: {key}"
    # neutral-gate sentinel policy checks (docs/PHASE2_SPEC.md); note
    # recruiter_response_rate's schema minimum is 0 (no -1 "missing" sentinel
    # allowed, unlike github_activity_score/offer_acceptance_rate), so 0.5 is
    # the schema-valid neutral midpoint used instead.
    assert sig["recruiter_response_rate"] == 0.5
    assert sig["interview_completion_rate"] == 0.5
    assert sig["profile_completeness_score"] == 50
    assert sig["skill_assessment_scores"] == {}
    assert sig["open_to_work_flag"] is True
    assert sig["last_active_date"] == REF_DATE
    # GitHub URL is present in the resume -> no longer the "unknown" -1 sentinel
    assert sig["github_activity_score"] != -1


def test_github_absent_keeps_neutral_sentinel():
    text_no_github = RESUME.replace("github.com/xyzcandidate", "").replace(
        "github.com/xyzcandidate/ratelimiter", "")
    res = compile_resume(text_no_github, MockBackend(_good), ref_date=REF_DATE)
    assert res.candidate["redrob_signals"]["github_activity_score"] == -1


def test_useless_model_degrades_to_valid_sentinels():
    res = compile_resume(RESUME, MockBackend(_bad), ref_date=REF_DATE)
    assert res.validation.ok, res.validation.error
    assert res.health["defaulted"], "should record defaulted blocks"
    cand = res.candidate
    assert cand["career_history"], "must sentinel at least one career_history entry"
    assert cand["profile"]["anonymized_name"]


def test_rate_limited_backend_still_produces_valid_candidate():
    res = compile_resume(RESUME, MockBackend(_rate_limited), ref_date=REF_DATE)
    assert res.validation.ok, res.validation.error
    assert res.health["defaulted"]


def test_logger_and_on_event_fire_per_leaf():
    log = HarnessLogger()
    events = []
    res = compile_resume(RESUME, MockBackend(_good), logger=log,
                         on_event=events.append, ref_date=REF_DATE)
    assert res.validation.ok, res.validation.error
    # profile, career_history, education, skills, projects >= 5 leaf calls
    assert len(log.entries) >= 5, [e.leaf for e in log.entries]
    assert len(events) == len(log.entries), "on_event fires once per recorded leaf"
    leaves = [e.leaf for e in log.entries]
    assert "profile" in leaves
    assert "career_history" in leaves
    assert "education" in leaves
    assert "skills" in leaves
    assert "projects" in leaves


def test_health_has_telemetry_summary():
    log = HarnessLogger()
    res = compile_resume(RESUME, MockBackend(_good), logger=log, ref_date=REF_DATE)
    assert "telemetry" in res.health
    tel = res.health["telemetry"]
    assert tel["total_calls"] == len(log.entries)


def test_candidate_yaml_and_json_roundtrip():
    import yaml
    res = compile_resume(RESUME, MockBackend(_good), ref_date=REF_DATE)
    loaded_yaml = yaml.safe_load(res.candidate_yaml)
    loaded_json = json.loads(res.candidate_json)
    assert loaded_yaml["candidate_id"] == res.candidate["candidate_id"]
    assert loaded_json["candidate_id"] == res.candidate["candidate_id"]


if __name__ == "__main__":
    for fn in [test_good_model_produces_valid_candidate,
               test_candidate_id_format,
               test_projects_land_in_summary_not_career_history,
               test_all_23_redrob_signals_present_with_sentinels,
               test_github_absent_keeps_neutral_sentinel,
               test_useless_model_degrades_to_valid_sentinels,
               test_rate_limited_backend_still_produces_valid_candidate,
               test_logger_and_on_event_fire_per_leaf,
               test_health_has_telemetry_summary,
               test_candidate_yaml_and_json_roundtrip]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all ok")
