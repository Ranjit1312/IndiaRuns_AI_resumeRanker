"""Offline tests for eval_study.compare (deterministic MockCompareBackend —
no API key, no network).

Two things are proven here:
  1. The metric functions (pairwise_accuracy, kendall_tau, stability_stddev)
     compute the correct values on small hand-built score matrices where the
     "gold" tier ordering and the answer are both known by inspection.
  2. The harness wires methods A/B/C end-to-end (discover_dataset -> run_study
     -> write_report/write_csv) using the synthetic labeled resume set and a
     deterministic mock backend, entirely offline.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_study.compare import (
    MockCompareBackend,
    Row,
    StudyResult,
    discover_dataset,
    kendall_tau,
    make_mock_backend,
    pairwise_accuracy,
    run_study,
    score_naive_llm,
    score_our_engine,
    score_rubric_llm,
    stability_stddev,
    write_csv,
    write_report,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# --------------------------------------------------------------------------- #
# metric unit tests — small synthetic score matrices, gold order strong>b>weak
# --------------------------------------------------------------------------- #
def test_pairwise_accuracy_perfect_order():
    scores = {"strong": 90.0, "borderline": 60.0, "weak": 20.0}
    assert pairwise_accuracy(scores) == 1.0


def test_pairwise_accuracy_fully_inverted():
    scores = {"strong": 10.0, "borderline": 50.0, "weak": 90.0}
    assert pairwise_accuracy(scores) == 0.0


def test_pairwise_accuracy_partial_order():
    # strong>weak (correct), strong>borderline (correct), borderline<weak (wrong)
    # gold pairs: (strong,borderline) correct, (strong,weak) correct,
    # (borderline,weak) WRONG since borderline(40) < weak(50)
    scores = {"strong": 90.0, "borderline": 40.0, "weak": 50.0}
    acc = pairwise_accuracy(scores)
    assert abs(acc - 2 / 3) < 1e-9


def test_pairwise_accuracy_ties_count_wrong():
    scores = {"strong": 50.0, "borderline": 50.0, "weak": 50.0}
    assert pairwise_accuracy(scores) == 0.0


def test_pairwise_accuracy_needs_two_tiers():
    assert pairwise_accuracy({"strong": 90.0}) is None
    assert pairwise_accuracy({}) is None


def test_kendall_tau_perfect_order():
    scores = {"strong": 90.0, "borderline": 60.0, "weak": 20.0}
    assert kendall_tau(scores) == 1.0


def test_kendall_tau_fully_inverted():
    scores = {"strong": 10.0, "borderline": 50.0, "weak": 90.0}
    assert kendall_tau(scores) == -1.0


def test_kendall_tau_partial_order():
    # same partial-order fixture as pairwise: 2 concordant, 1 discordant
    scores = {"strong": 90.0, "borderline": 40.0, "weak": 50.0}
    tau = kendall_tau(scores)
    assert abs(tau - (2 - 1) / (2 + 1)) < 1e-9


def test_kendall_tau_needs_two_tiers():
    assert kendall_tau({"strong": 90.0}) is None


def test_stability_stddev_zero_for_identical_scores():
    assert stability_stddev([50.0, 50.0, 50.0]) == 0.0


def test_stability_stddev_positive_for_varying_scores():
    dev = stability_stddev([40.0, 50.0, 60.0])
    assert dev > 0.0


def test_stability_stddev_single_value_is_zero():
    assert stability_stddev([42.0]) == 0.0


# --------------------------------------------------------------------------- #
# StudyResult aggregation, built from a hand-constructed row list (no I/O)
# --------------------------------------------------------------------------- #
def _hand_built_study() -> StudyResult:
    rows = []
    # method A: perfect ranking, deterministic (stddev 0) across repeats
    for rep in range(3):
        rows.append(Row("jd1", "strong", "A_our_engine", rep, 90.0, None, 0.01, None))
        rows.append(Row("jd1", "borderline", "A_our_engine", rep, 60.0, None, 0.01, None))
        rows.append(Row("jd1", "weak", "A_our_engine", rep, 20.0, None, 0.01, None))
    # method C: inverted ranking, with jitter across repeats (nonzero stddev)
    c_scores = {"strong": [10.0, 12.0, 8.0], "borderline": [50.0, 48.0, 52.0],
               "weak": [90.0, 88.0, 92.0]}
    for tier, vals in c_scores.items():
        for rep, v in enumerate(vals):
            rows.append(Row("jd1", tier, "C_naive_llm", rep, v, 100, 0.5, None))
    return StudyResult(rows=rows, repeats=3, pairs=[])


def test_study_ranking_metrics_perfect_vs_inverted():
    study = _hand_built_study()
    a = study.ranking_metrics("A_our_engine")
    c = study.ranking_metrics("C_naive_llm")
    assert a["pairwise_accuracy"] == 1.0
    assert a["kendall_tau"] == 1.0
    assert c["pairwise_accuracy"] == 0.0
    assert c["kendall_tau"] == -1.0


def test_study_stability_by_deterministic_vs_jittered():
    study = _hand_built_study()
    a_stab = study.stability_by("A_our_engine")
    c_stab = study.stability_by("C_naive_llm")
    assert a_stab == 0.0
    assert c_stab > 0.0


def test_study_cost_latency_aggregates():
    study = _hand_built_study()
    cl = study.cost_latency("C_naive_llm")
    assert cl["n_calls"] == 9
    assert cl["total_tokens"] == 900
    assert cl["n_errors"] == 0


# --------------------------------------------------------------------------- #
# dataset discovery — the real synthetic-resume set on disk
# --------------------------------------------------------------------------- #
def test_discover_dataset_finds_labeled_pairs():
    dataset = discover_dataset()
    assert len(dataset) >= 4   # at least the 4 JDs x >=1 tier documented in INDEX.md
    slugs_tiers = {(p.jd_slug, p.tier) for p in dataset}
    assert ("stripe_backend-software-engineer", "strong") in slugs_tiers
    assert ("stripe_backend-software-engineer", "weak") in slugs_tiers
    for p in dataset:
        assert os.path.isfile(p.resume_path)
        assert os.path.isfile(p.jd_txt_path)
        assert os.path.isfile(p.jd_profile_path)


def test_discover_dataset_respects_limit():
    dataset = discover_dataset(limit=2)
    assert len(dataset) == 2


# --------------------------------------------------------------------------- #
# end-to-end offline wiring: mock backend, one JD's 3 tiers, methods A/B/C
# --------------------------------------------------------------------------- #
def test_mock_backend_no_network_attrs():
    backend = make_mock_backend()
    assert isinstance(backend, MockCompareBackend)
    assert hasattr(backend, "generate") and callable(backend.generate)
    assert hasattr(backend, "embed") and callable(backend.embed)


def test_score_rubric_and_naive_llm_offline():
    backend = make_mock_backend()
    dataset = discover_dataset()
    pair = next(p for p in dataset if p.jd_slug == "stripe_backend-software-engineer"
                and p.tier == "strong")
    jd_text = open(pair.jd_txt_path, encoding="utf-8").read()
    resume_text = open(pair.resume_path, encoding="utf-8").read()

    b = score_rubric_llm(jd_text, resume_text, backend)
    c = score_naive_llm(jd_text, resume_text, backend)
    assert b.error is None and c.error is None
    assert 0.0 <= b.score <= 100.0
    assert 0.0 <= c.score <= 100.0
    assert backend.calls >= 2   # both methods actually called generate()


def test_score_our_engine_offline_end_to_end():
    backend = make_mock_backend()
    dataset = discover_dataset()
    pair = next(p for p in dataset if p.jd_slug == "stripe_backend-software-engineer"
                and p.tier == "strong")
    resume_text = open(pair.resume_path, encoding="utf-8").read()

    res = score_our_engine(pair.jd_profile_path, resume_text, backend,
                           ref_date="2026-06-06")
    assert res.error is None, f"method A failed: {res.error}"
    assert 0.0 <= res.score <= 100.0


def test_run_study_wires_a_b_c_without_network():
    backend = make_mock_backend()
    dataset = [p for p in discover_dataset()
               if p.jd_slug == "stripe_backend-software-engineer"]
    assert len(dataset) == 3   # strong/borderline/weak

    study = run_study(dataset, backend, repeats=2)
    methods_seen = {r.method for r in study.rows}
    assert methods_seen == {"A_our_engine", "B_rubric_llm", "C_naive_llm"}
    # 3 pairs x 3 methods x 2 repeats
    assert len(study.rows) == 18
    assert all(0.0 <= r.score <= 100.0 for r in study.rows)


def test_run_study_report_and_csv_written(tmp_path):
    backend = make_mock_backend()
    dataset = [p for p in discover_dataset()
               if p.jd_slug == "stripe_backend-software-engineer"]
    study = run_study(dataset, backend, repeats=2)

    out_dir = str(tmp_path / "eval_study_out")
    report_path, csv_path = write_report(study, out_dir=out_dir)

    assert os.path.isfile(report_path)
    assert os.path.isfile(csv_path)

    report_text = open(report_path, encoding="utf-8").read()
    assert "synthetic data with known labels" in report_text
    assert "Our engine" in report_text
    assert "strong" in report_text and "borderline" in report_text and "weak" in report_text

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 18
    assert {"jd_slug", "tier", "method", "repeat", "score", "tokens",
            "elapsed_s", "error"} <= set(reader.fieldnames)


def test_write_csv_standalone(tmp_path):
    study = _hand_built_study()
    out_path = str(tmp_path / "scores.csv")
    write_csv(study, out_path)
    assert os.path.isfile(out_path)
    with open(out_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(study.rows)
