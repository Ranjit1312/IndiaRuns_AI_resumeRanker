"""eval_study — comparison harness: OUR engine vs naive single-LLM baselines.

Public surface:
    score_our_engine(jd_slug, resume_text, backend, *, ref_date=None) -> ScoreResult
    score_rubric_llm(jd_text, resume_text, backend) -> ScoreResult
    score_naive_llm(jd_text, resume_text, backend) -> ScoreResult
    run_study(dataset, backend, *, repeats=3) -> StudyResult
    write_report(study, out_dir) -> (report_path, csv_path)

See eval_study/compare.py module docstring for the full A/B/C method
definitions and the metrics computed against the synthetic gold tiers.
"""
from __future__ import annotations

from .compare import (
    ScoreResult,
    StudyResult,
    discover_dataset,
    kendall_tau,
    pairwise_accuracy,
    run_study,
    score_naive_llm,
    score_our_engine,
    score_rubric_llm,
    write_report,
)

__all__ = [
    "ScoreResult", "StudyResult", "discover_dataset",
    "score_our_engine", "score_rubric_llm", "score_naive_llm",
    "pairwise_accuracy", "kendall_tau", "run_study", "write_report",
]
