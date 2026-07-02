"""Offline harness tests (MockBackend — no API key, no network).

Proves the RLM harness produces a schema-valid jd_profile even when the model is
perfect AND when it is useless (graceful sentinel degradation), and that bad
regexes/ids from a model are sanitized rather than fatal.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.backends import MockBackend
from harness.coerce import compile_jd
from harness.jsonutil import extract_json

JD = """Senior Data Engineer — Acme
About the role
Build and operate batch and streaming data pipelines that power analytics.
Responsibilities
- Design ETL with Spark and Airflow
- Own data quality and reliability
Basic Qualifications
- 4+ years of data engineering experience
- Strong Python and SQL
Preferred Qualifications
- Experience with cloud data warehouses
Location: Bangalore (hybrid)
"""


def _good(prompt, system):
    if '"title"' in prompt:
        return json.dumps({"title": "Senior Data Engineer", "company": "Acme",
                           "domain": "data engineering", "summary": "Build data pipelines.",
                           "min_years": 4, "max_years": 9, "peak_years": 6,
                           "sigma_years": 2.5, "notice_preference_days": 45})
    if '"preferred"' in prompt:
        return json.dumps({"preferred": ["bangalore"], "acceptable": ["pune"],
                           "relocation_acceptable": True, "remote_acceptable": False})
    if "MUST-HAVE capability" in prompt:
        return json.dumps(["data pipelines", "spark", "sql", "cloud data warehouse"])
    if "For the capability" in prompt:
        return json.dumps({"query": "building batch and streaming data pipelines",
                           "evidence_regex": "airflow|spark|kafka|etl", "dense_weight": 0.2})
    if "IN-domain" in prompt:
        return json.dumps({"in_domain_terms": ["data engineering", "etl", "spark"],
                           "out_of_domain_terms": ["frontend", "marketing"],
                           "in_domain_regex": "data engineer|etl|spark",
                           "out_of_domain_regex": "frontend|marketing"})
    if "relevant_skill_regex" in prompt:
        return json.dumps({"relevant_skill_regex": "python|sql|spark|airflow|aws"})
    if "supported red-flag gates" in prompt:
        return json.dumps({"cv_primary": False, "job_hopper": True,
                           "only_consulting": False, "stale_ic_role": True})
    if "coaching sidecar" in prompt:
        return json.dumps({"hard_requirements": [{"text": "4+ years", "kind": "yoe"}],
                           "explicit_exclusions": [{"text": "no frontend-only backgrounds"}],
                           "coaching_notes": "Show pipeline scale and reliability."})
    return "{}"


def _bad(prompt, system):
    return "Sorry, I can't help with that."   # never valid JSON


def _bad_regex(prompt, system):
    if "For the capability" in prompt:      # unbalanced regex must be dropped to null
        return json.dumps({"query": "x", "evidence_regex": "spark(|airflow", "dense_weight": 0.2})
    return _good(prompt, system)


def test_good_model_produces_valid_profile():
    res = compile_jd(JD, MockBackend(_good))
    assert res.validation.ok, res.validation.error
    ids = [s["id"] for s in res.profile["signals"]]
    assert len(ids) == len(set(ids)) >= 3
    assert set(res.profile["red_flags"]) == {"cv_primary", "job_hopper",
                                             "only_consulting", "stale_ic_role"}
    assert res.profile["role"]["notice_preference_days"] == 45
    assert res.meta["must_haves"], "coaching meta should link signals"


def test_useless_model_degrades_to_valid_sentinels():
    res = compile_jd(JD, MockBackend(_bad))
    assert res.validation.ok, res.validation.error      # still valid!
    assert res.health["defaulted"], "should record defaulted blocks"
    assert res.profile["signals"], "must have >=1 signal"


def test_bad_regex_is_sanitized_not_fatal():
    res = compile_jd(JD, MockBackend(_bad_regex))
    assert res.validation.ok, res.validation.error
    for s in res.profile["signals"]:
        er = s["evidence_regex"]
        assert er is None or _compiles(er)


def test_jsonutil_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('Here you go: {"b": [1,2]} — done')["b"] == [1, 2]


def _compiles(rx):
    import re
    try:
        re.compile(rx)
        return True
    except re.error:
        return False


if __name__ == "__main__":
    for fn in [test_good_model_produces_valid_profile,
               test_useless_model_degrades_to_valid_sentinels,
               test_bad_regex_is_sanitized_not_fatal,
               test_jsonutil_handles_fences_and_prose]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all ok")
