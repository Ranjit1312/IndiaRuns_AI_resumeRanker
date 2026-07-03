"""compare.py — OUR engine vs naive single-LLM baselines, on the LABELED
synthetic set (Phase 2 comparison study).

Dataset: `data/synthetic_resumes/<jd_slug>_<tier>/resume.txt`, tier in
{strong, borderline, weak}, seeded off each JD's gold `jd_profile.yaml` /
`jd_meta.yaml` (see `data/synthetic_resumes/INDEX.md` and
`tools/gen_resumes.py`). The folder name ENCODES the gold label — this is a
synthetic, labeled sanity set, not a claim about real-world hire quality (the
report emits this disclaimer verbatim; do not strip it).

Three scoring methods, each mapping a (JD, resume) pair -> a 0-100 fit score:

  A) OUR ENGINE (`score_our_engine`)
     Loads the JD's gold `jd_profile.yaml` + the shared `jd/method_config.yaml`
     via `redrob_ranker.profile.load` (temp-file pattern, see
     `harness/validate.py::validate_profile_dict`), parses the resume via
     `harness.resume.compile_resume` -> candidate dict, then scores with
     `redrob_ranker.fit.score_candidate(candidate, profile, method, backend)`.
     This is STRUCTURED and INTERPRETABLE: the 0-100 `FitResult.overall` is an
     explicit function of per-signal dense/lexical/evidence channels and named
     multiplicative gates (integrity, availability, notice, location, red
     flags) — see `redrob_ranker/fit.py`'s module + `FitResult` docstrings.
     Given a fixed embedding backend, it is DETERMINISTIC (no sampling) and
     produces machine-readable `gaps`/`red_flags`/`per_signal` alongside the
     score. It costs TWO structured extraction passes (JD compile is assumed
     pre-done/cached here since we score against the *gold* jd_profile.yaml,
     not a freshly-compiled one; the resume is compiled fresh per candidate)
     plus one embedding call.

  B) SINGLE-LLM WITH RUBRIC (`score_rubric_llm`)
     ONE `backend.generate` call. The prompt contains the raw JD text, the raw
     resume text, and a short explicit scoring rubric (experience fit, skill/
     domain match, evidence of impact, availability signals) and asks for
     strict JSON `{"fit_0_100": <int>, "reasons": [...]}`. This is a single
     OPAQUE model call: the score is whatever the model decides in one shot,
     it can drift run-to-run (temperature-sampled), and the "reasons" are
     free-text model output, not derived from any auditable per-signal math.

  C) NAIVE COPY-PASTE LLM (`score_naive_llm`)
     ONE `backend.generate` call with a MINIMAL prompt ("Given this JD and
     resume, rate fit 0-100") and the raw JD/resume text pasted in — no
     rubric, no requested structure beyond asking for a number. This is the
     truly naive baseline a non-technical user would improvise with a chatbot.

A vs B/C, in one line: A is a structured, interpretable, (embeddings-)
deterministic pipeline with an explicit scoring function; B and C are each a
single opaque LLM call whose score is not decomposable and can vary between
identical runs.

Metrics (per method, computed against the KNOWN tiers strong > borderline >
weak within each JD):
  - Ranking accuracy: pairwise-correct fraction (does the method's score order
    agree with the gold tier order, for every pair within a JD?) and
    Kendall's tau vs the gold rank, both aggregated (mean) across JDs.
  - Stability: run each method N times (default 3) per (JD, resume) pair;
    report the score std-dev, mean across pairs. Method A is deterministic
    given a fixed embedding backend (embeddings are not temperature-sampled),
    so its stability is expected to be at/near 0; B/C use a sampled
    `backend.generate` call and can drift.
  - Cost/latency: total tokens (`backend.last_usage`, when the backend
    exposes it) and wall-clock time, summed per method across the whole run.

Output: `data/eval_study/comparison_report.md` (human-readable) and
`data/eval_study/scores.csv` (raw per-pair-per-method-per-repeat rows).

CLI:
    ./.venv/Scripts/python -m eval_study.compare --key $GOOGLE_API_KEY \\
        --models gemma-4-26b-a4b-it [--repeats 3] [--limit N]
    ./.venv/Scripts/python -m eval_study.compare --mock   # offline, no key
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import yaml

from harness.jsonutil import extract_json
from harness.resume import compile_resume
from harness.validate import METHOD_PATH, validate_profile_dict
from redrob_ranker import profile as rprofile
from redrob_ranker.fit import score_candidate

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
EVAL_JDS_DIR = os.path.join(_ROOT, "data", "eval_jds")
SYNTHETIC_RESUMES_DIR = os.path.join(_ROOT, "data", "synthetic_resumes")
REPORT_DIR = os.path.join(_ROOT, "data", "eval_study")

TIERS = ["strong", "borderline", "weak"]                 # gold rank, best first
TIER_RANK = {t: i for i, t in enumerate(TIERS)}           # 0 = best
METHODS = ["A_our_engine", "B_rubric_llm", "C_naive_llm"]

_FOLDER_RE = re.compile(r"^(?P<slug>.+)_(?P<tier>strong|borderline|weak)$")

REF_DATE_DEFAULT = "2026-06-06"   # matches jd/method_config.yaml runtime.ref_date


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    jd_slug: str
    tier: str
    resume_path: str
    jd_txt_path: str
    jd_profile_path: str


# Canonical layout used as the plain-text source for each (jd_slug, tier)
# pair. `data/synthetic_resumes/INDEX.md`'s "Layout matrix" renders every
# (jd_slug, tier) in several layouts (single_column, two_column, table_heavy,
# ats_plain, image_only) with IDENTICAL underlying content/PII/seed — only
# the rendering differs (see tools/resume_layouts.py). single_column is the
# clean, layout-neutral baseline, so it's what this study scores against;
# it is not a claim that layout-robustness itself is being measured here.
CANONICAL_LAYOUT = "single_column"


def discover_dataset(resumes_dir: str = SYNTHETIC_RESUMES_DIR,
                      eval_jds_dir: str = EVAL_JDS_DIR,
                      limit: "int | None" = None,
                      layout: str = CANONICAL_LAYOUT) -> list:
    """Find every `<jd_slug>_<tier>/<layout>/resume.txt` with a matching gold JD.

    Only folders whose `jd_slug` has BOTH a `jd.txt` and a `jd_profile.yaml`
    under `data/eval_jds/<jd_slug>/` are included (method A needs the gold
    profile; B/C need the raw JD text). Falls back to `<jd_slug>_<tier>/
    resume.txt` directly (the older flat layout) if the layout subfolder
    isn't present, so this keeps working against either resume-set shape.
    """
    # NOTE: glob.escape() is required because this repo's path contains a
    # literal "[PUB]" segment — glob() would otherwise treat "[PUB]" as a
    # character class and silently match nothing.
    pairs: list[Pair] = []
    for d in sorted(glob.glob(os.path.join(glob.escape(resumes_dir), "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = _FOLDER_RE.match(name)
        if not m:
            continue
        slug, tier = m.group("slug"), m.group("tier")
        resume_path = os.path.join(d, layout, "resume.txt")
        if not os.path.isfile(resume_path):
            resume_path = os.path.join(d, "resume.txt")   # flat-layout fallback
        jd_txt_path = os.path.join(eval_jds_dir, slug, "jd.txt")
        jd_profile_path = os.path.join(eval_jds_dir, slug, "jd_profile.yaml")
        if not (os.path.isfile(resume_path) and os.path.isfile(jd_txt_path)
                and os.path.isfile(jd_profile_path)):
            continue
        pairs.append(Pair(jd_slug=slug, tier=tier, resume_path=resume_path,
                          jd_txt_path=jd_txt_path, jd_profile_path=jd_profile_path))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


# --------------------------------------------------------------------------- #
# scoring methods A / B / C
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    score: float                 # 0-100
    reasons: list = field(default_factory=list)
    tokens: "int | None" = None
    elapsed_s: float = 0.0
    error: "str | None" = None


def score_our_engine(jd_profile_path: str, resume_text: str, backend, *,
                      method_path: str = METHOD_PATH,
                      ref_date: "str | None" = None) -> ScoreResult:
    """Method A: gold jd_profile.yaml + method_config.yaml -> Profile/Method
    (redrob_ranker.profile.load), resume -> candidate (compile_resume), then
    redrob_ranker.fit.score_candidate. See module docstring."""
    t0 = time.monotonic()
    try:
        profile, method = rprofile.load(jd_profile_path, method_path)
        resume_res = compile_resume(resume_text, backend, ref_date=ref_date)
        fit = score_candidate(resume_res.candidate, profile, method, backend,
                              ref_date=ref_date)
        return ScoreResult(score=float(fit.overall), reasons=list(fit.gaps),
                           elapsed_s=time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 — never crash the study on one pair
        return ScoreResult(score=0.0, error=str(exc), elapsed_s=time.monotonic() - t0)


_RUBRIC_PROMPT = """You are scoring how well a candidate's resume fits a job description.

Use this rubric:
1. Experience level fit (years of experience vs role seniority)
2. Skill / domain match (do the candidate's skills and career history match the JD's core requirements?)
3. Evidence of impact (specific, quantified accomplishments vs vague claims)
4. Availability / practical fit (location, notice period, if mentioned)

Job description:
---
{jd_text}
---

Resume:
---
{resume_text}
---

Return ONLY valid JSON in exactly this shape:
{{"fit_0_100": <integer 0-100>, "reasons": ["<short reason 1>", "<short reason 2>", "..."]}}
"""

_NAIVE_PROMPT = """Given this JD and resume, rate fit 0-100.

JD:
{jd_text}

Resume:
{resume_text}
"""


def _parse_llm_score(raw: str) -> "tuple[float, list]":
    """Best-effort: structured JSON first, else the first 0-100 number in text."""
    try:
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            score = parsed.get("fit_0_100", parsed.get("fit", parsed.get("score")))
            reasons = parsed.get("reasons") or []
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            if score is not None:
                return float(np.clip(float(score), 0.0, 100.0)), reasons
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", raw or "")
    if m:
        return float(np.clip(float(m.group(1)), 0.0, 100.0)), []
    return 0.0, []


def score_rubric_llm(jd_text: str, resume_text: str, backend) -> ScoreResult:
    """Method B: ONE backend.generate call, JD + resume + a short rubric,
    JSON-structured output. Single opaque call — see module docstring."""
    t0 = time.monotonic()
    prompt = _RUBRIC_PROMPT.format(jd_text=jd_text, resume_text=resume_text)
    try:
        raw = backend.generate(prompt, temperature=0.2, max_tokens=512)
        score, reasons = _parse_llm_score(raw)
        usage = getattr(backend, "last_usage", None) or {}
        tokens = usage.get("total_tokens")
        return ScoreResult(score=score, reasons=reasons, tokens=tokens,
                           elapsed_s=time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return ScoreResult(score=0.0, error=str(exc), elapsed_s=time.monotonic() - t0)


def score_naive_llm(jd_text: str, resume_text: str, backend) -> ScoreResult:
    """Method C: ONE backend.generate call, minimal prompt, no rubric/structure
    — the truly naive copy-paste-into-a-chatbot baseline. See module docstring."""
    t0 = time.monotonic()
    prompt = _NAIVE_PROMPT.format(jd_text=jd_text, resume_text=resume_text)
    try:
        raw = backend.generate(prompt, temperature=0.2, max_tokens=256)
        score, reasons = _parse_llm_score(raw)
        usage = getattr(backend, "last_usage", None) or {}
        tokens = usage.get("total_tokens")
        return ScoreResult(score=score, reasons=reasons, tokens=tokens,
                           elapsed_s=time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return ScoreResult(score=0.0, error=str(exc), elapsed_s=time.monotonic() - t0)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pairwise_accuracy(scores_by_tier: dict) -> "float | None":
    """Fraction of {strong,borderline,weak} pairs within one JD where the
    higher-gold-tier resume also got the higher score (ties count as wrong).

    `scores_by_tier`: {tier: score}. Returns None if fewer than 2 tiers present.
    """
    tiers = [t for t in TIERS if t in scores_by_tier]
    if len(tiers) < 2:
        return None
    correct, total = 0, 0
    for a, b in combinations(tiers, 2):
        total += 1
        # a comes before b in TIERS => a is the better gold tier
        if scores_by_tier[a] > scores_by_tier[b]:
            correct += 1
    return correct / total if total else None


def kendall_tau(scores_by_tier: dict) -> "float | None":
    """Kendall's tau between the method's score rank and the gold tier rank,
    for the tiers present in `scores_by_tier`. Returns None if fewer than 2
    tiers present (tau is undefined for n<2)."""
    tiers = [t for t in TIERS if t in scores_by_tier]
    n = len(tiers)
    if n < 2:
        return None
    concordant, discordant = 0, 0
    for a, b in combinations(tiers, 2):
        gold_order = TIER_RANK[a] - TIER_RANK[b]       # a is better gold tier => negative
        score_order = scores_by_tier[a] - scores_by_tier[b]
        if gold_order == 0 or score_order == 0:
            continue
        # gold better (gold_order<0) should mean score higher (score_order>0)
        same_direction = (gold_order < 0) == (score_order > 0)
        if same_direction:
            concordant += 1
        else:
            discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def stability_stddev(repeat_scores: list) -> float:
    """Std-dev of a list of repeated scores for one (JD, resume, method)."""
    if len(repeat_scores) < 2:
        return 0.0
    return float(statistics.pstdev(repeat_scores))


# --------------------------------------------------------------------------- #
# run the study
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    jd_slug: str
    tier: str
    method: str
    repeat: int
    score: float
    tokens: "int | None"
    elapsed_s: float
    error: "str | None"


@dataclass
class StudyResult:
    rows: list                       # list[Row] — every (pair, method, repeat)
    repeats: int
    pairs: list                      # list[Pair]

    def scores_by(self, method: str) -> dict:
        """{jd_slug: {tier: mean_score}} for one method, averaged over repeats."""
        out: dict = {}
        for r in self.rows:
            if r.method != method:
                continue
            out.setdefault(r.jd_slug, {}).setdefault(r.tier, []).append(r.score)
        return {slug: {tier: float(np.mean(vals)) for tier, vals in tiers.items()}
                for slug, tiers in out.items()}

    def stability_by(self, method: str) -> "float | None":
        """Mean std-dev across all (jd_slug, tier) pairs for one method."""
        groups: dict = {}
        for r in self.rows:
            if r.method != method:
                continue
            groups.setdefault((r.jd_slug, r.tier), []).append(r.score)
        devs = [stability_stddev(v) for v in groups.values() if len(v) >= 2]
        return float(np.mean(devs)) if devs else None

    def ranking_metrics(self, method: str) -> dict:
        """Mean pairwise accuracy + mean Kendall-tau across JDs, for one method."""
        by_jd = self.scores_by(method)
        accs, taus = [], []
        for slug, tier_scores in by_jd.items():
            acc = pairwise_accuracy(tier_scores)
            tau = kendall_tau(tier_scores)
            if acc is not None:
                accs.append(acc)
            if tau is not None:
                taus.append(tau)
        return {
            "pairwise_accuracy": float(np.mean(accs)) if accs else None,
            "kendall_tau": float(np.mean(taus)) if taus else None,
            "n_jds": len(by_jd),
        }

    def cost_latency(self, method: str) -> dict:
        rows = [r for r in self.rows if r.method == method]
        total_tokens = sum(r.tokens for r in rows if r.tokens is not None)
        n_with_tokens = sum(1 for r in rows if r.tokens is not None)
        total_elapsed = sum(r.elapsed_s for r in rows)
        n_errors = sum(1 for r in rows if r.error)
        return {
            "total_tokens": total_tokens if n_with_tokens else None,
            "total_elapsed_s": round(total_elapsed, 3),
            "n_calls": len(rows),
            "n_errors": n_errors,
        }


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_study(dataset: list, backend, *, repeats: int = 3,
              method_path: str = METHOD_PATH,
              ref_date: str = REF_DATE_DEFAULT,
              methods: "list[str] | None" = None,
              on_row=None) -> StudyResult:
    """Score every (JD, resume) pair in `dataset` with every method in
    `methods` (default all of A/B/C), `repeats` times each. `backend` must
    expose `.generate()` (all methods) and `.embed()` (method A only, via
    `redrob_ranker.fit.score_candidate`'s embedding requirement).

    `on_row(Row)` is called after each individual score, if given (for a
    CLI progress print) — never required.
    """
    methods = methods or METHODS
    rows: list = []
    for pair in dataset:
        jd_text = _read(pair.jd_txt_path)
        resume_text = _read(pair.resume_path)
        for rep in range(repeats):
            if "A_our_engine" in methods:
                res = score_our_engine(pair.jd_profile_path, resume_text, backend,
                                       method_path=method_path, ref_date=ref_date)
                row = Row(pair.jd_slug, pair.tier, "A_our_engine", rep, res.score,
                          res.tokens, res.elapsed_s, res.error)
                rows.append(row)
                if on_row:
                    on_row(row)
            if "B_rubric_llm" in methods:
                res = score_rubric_llm(jd_text, resume_text, backend)
                row = Row(pair.jd_slug, pair.tier, "B_rubric_llm", rep, res.score,
                          res.tokens, res.elapsed_s, res.error)
                rows.append(row)
                if on_row:
                    on_row(row)
            if "C_naive_llm" in methods:
                res = score_naive_llm(jd_text, resume_text, backend)
                row = Row(pair.jd_slug, pair.tier, "C_naive_llm", rep, res.score,
                          res.tokens, res.elapsed_s, res.error)
                rows.append(row)
                if on_row:
                    on_row(row)
    return StudyResult(rows=rows, repeats=repeats, pairs=dataset)


# --------------------------------------------------------------------------- #
# output: scores.csv + comparison_report.md
# --------------------------------------------------------------------------- #
_METHOD_LABEL = {
    "A_our_engine": "A) Our engine (structured, interpretable)",
    "B_rubric_llm": "B) Single-LLM w/ rubric (one opaque call)",
    "C_naive_llm": "C) Naive copy-paste LLM (one opaque call, no structure)",
}


def write_csv(study: StudyResult, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["jd_slug", "tier", "method", "repeat", "score",
                   "tokens", "elapsed_s", "error"])
        for r in study.rows:
            w.writerow([r.jd_slug, r.tier, r.method, r.repeat,
                       f"{r.score:.4f}", r.tokens if r.tokens is not None else "",
                       f"{r.elapsed_s:.4f}", r.error or ""])
    return out_path


def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_report_md(study: StudyResult, methods: "list[str] | None" = None) -> str:
    methods = methods or METHODS
    lines = []
    lines.append("# Comparison study: our engine vs naive single-LLM baselines")
    lines.append("")
    lines.append(
        "**Computed on synthetic data with known labels.** Every (JD, resume) "
        "pair below comes from `data/synthetic_resumes/<jd_slug>_<tier>/resume.txt`, "
        "where `tier` in `{strong, borderline, weak}` is a GOLD LABEL baked into "
        "the folder name (not inferred) — the resumes were generated by "
        "`tools/gen_resumes.py`, seeded off each JD's own gold "
        "`jd_profile.yaml`/`jd_meta.yaml`, to be a realistic but synthetic and "
        "PII-free spread of strong/borderline/weak fits. Numbers here measure "
        "each method's agreement with that KNOWN ordering — they are not a "
        "claim about real-world hiring accuracy.")
    lines.append("")
    lines.append("## Tier definitions")
    lines.append("")
    lines.append("| Tier | Meaning |")
    lines.append("|---|---|")
    lines.append("| strong | Deep, recent, on-domain experience; expected to score highest for its JD |")
    lines.append("| borderline | Partial / adjacent fit; expected to score in the middle |")
    lines.append("| weak | Off-domain or junior/unrelated experience; expected to score lowest |")
    lines.append("")
    lines.append(
        "Gold order per JD: **strong > borderline > weak**. See "
        "`data/synthetic_resumes/INDEX.md` for the exact folder list.")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(
        "- **A) Our engine** — gold `jd_profile.yaml` + `jd/method_config.yaml` "
        "-> `redrob_ranker.profile.load`; resume -> `harness.resume.compile_resume` "
        "-> candidate; `redrob_ranker.fit.score_candidate(...)` -> `FitResult.overall` "
        "(0-100). Structured: an explicit, auditable function of per-signal dense/"
        "lexical/evidence channels and named multiplicative gates. Deterministic "
        "given a fixed embedding backend.")
    lines.append(
        "- **B) Single-LLM w/ rubric** — ONE `backend.generate` call with the raw "
        "JD text + raw resume text + a short scoring rubric -> JSON "
        "`{fit_0_100, reasons}`. A single opaque model call; can drift run-to-run.")
    lines.append(
        "- **C) Naive copy-paste LLM** — ONE `backend.generate` call, minimal "
        "prompt (\"Given this JD and resume, rate fit 0-100\"), raw text, no "
        "rubric or structure. The baseline a non-technical user would improvise.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | Pairwise accuracy | Kendall tau | Stability (std-dev) | Total tokens | Total wall-time (s) | Calls | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        rm = study.ranking_metrics(m)
        stab = study.stability_by(m)
        cl = study.cost_latency(m)
        lines.append(
            f"| {_METHOD_LABEL.get(m, m)} | {_fmt(rm['pairwise_accuracy'])} "
            f"| {_fmt(rm['kendall_tau'])} | {_fmt(stab)} "
            f"| {_fmt(cl['total_tokens'], 0)} | {_fmt(cl['total_elapsed_s'])} "
            f"| {cl['n_calls']} | {cl['n_errors']} |")
    lines.append("")
    lines.append(
        f"N JDs evaluated: {len(set(p.jd_slug for p in study.pairs))}. "
        f"N (JD, resume) pairs: {len(study.pairs)}. Repeats per pair per method: "
        f"{study.repeats}.")
    lines.append("")
    lines.append("## Per-JD scores (mean over repeats)")
    lines.append("")
    for m in methods:
        lines.append(f"### {_METHOD_LABEL.get(m, m)}")
        lines.append("")
        lines.append("| JD | strong | borderline | weak | pairwise acc | kendall tau |")
        lines.append("|---|---|---|---|---|---|")
        by_jd = study.scores_by(m)
        for slug in sorted(by_jd):
            ts = by_jd[slug]
            acc = pairwise_accuracy(ts)
            tau = kendall_tau(ts)
            lines.append(
                f"| {slug} | {_fmt(ts.get('strong'))} | {_fmt(ts.get('borderline'))} "
                f"| {_fmt(ts.get('weak'))} | {_fmt(acc)} | {_fmt(tau)} |")
        lines.append("")
    lines.append(
        "_Report generated by `eval_study/compare.py` — no numbers on this page "
        "are hardcoded; re-run `python -m eval_study.compare` to regenerate._")
    lines.append("")
    return "\n".join(lines)


def write_report(study: StudyResult, out_dir: str = REPORT_DIR,
                 methods: "list[str] | None" = None) -> "tuple[str, str]":
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "comparison_report.md")
    csv_path = os.path.join(out_dir, "scores.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report_md(study, methods=methods))
    write_csv(study, csv_path)
    return report_path, csv_path


# --------------------------------------------------------------------------- #
# tiny regex-based resume parsing used ONLY by MockCompareBackend, so the
# offline --mock run gives method A a real (not empty-stub) candidate to
# score. Matches the fixed layout emitted by tools/gen_resumes.py /
# tools/resume_layouts.py's "single_column" / "ats_plain" renderers:
#   SUMMARY / EXPERIENCE (Title, Company (X years, ...)\n- bullet...) /
#   PROJECTS / EDUCATION / SKILLS / "Notice period: N days|months"
# --------------------------------------------------------------------------- #
_RESUME_BLOCK_RE = re.compile(r"RESUME:\s*\n(.*?)\n\s*JSON", re.DOTALL)
_EXP_ENTRY_RE = re.compile(
    r"^(?P<title>[^,\n]+(?:,\s*[^,\n]*?(?:\([^)\n]*\))?[^,\n]*?)?),\s*"
    r"(?P<company>[^,\n]+?)\s*"
    r"\((?P<yrs>[\d.]+)\s*years?,\s*(?P<recency>Present|\+[\d.]+y)\)\s*\n"
    r"(?P<bullets>(?:-.*\n?)*)",
    re.MULTILINE)


def _extract_resume_block(prompt: str) -> str:
    """Pull the raw resume text out of a resume_prompts.py leaf prompt (each
    embeds it verbatim after a "RESUME:" marker, before the trailing "JSON:"
    cue) — falls back to the whole prompt if the marker isn't found."""
    m = _RESUME_BLOCK_RE.search(prompt)
    return m.group(1) if m else prompt


def _add_months(y: int, m: int, delta: int) -> "tuple[int, int]":
    total = (y * 12 + (m - 1)) - delta
    return total // 12, total % 12 + 1


def _mock_career_history(text: str, ref_date: str = REF_DATE_DEFAULT) -> list:
    """Walk entries in résumé order (most-recent-first, matching
    tools/gen_resumes.py's emission order) backward from `ref_date`, so
    start/end/duration_months are internally consistent — an inconsistent
    fabrication here would trip `redrob_ranker.fit.score_candidate`'s
    integrity ladder (career_sum vs yoe, single-role-vs-yoe checks) and crush
    `overall` uniformly regardless of tier, masking the real tier signal."""
    ref_y, ref_m, _ = (int(x) for x in ref_date.split("-"))
    out = []
    cursor_y, cursor_m = ref_y, ref_m   # end-of-current-role cursor, walks backward
    for m in _EXP_ENTRY_RE.finditer(text):
        yrs = float(m.group("yrs"))
        duration_months = max(1, round(yrs * 12))
        is_current = m.group("recency") == "Present"
        end_y, end_m = cursor_y, cursor_m
        start_y, start_m = _add_months(end_y, end_m, duration_months)
        bullets = " ".join(
            ln.strip("- ").strip() for ln in m.group("bullets").splitlines() if ln.strip())
        out.append({
            "company": m.group("company").strip(), "title": m.group("title").strip(),
            "start_date": f"{start_y:04d}-{start_m:02d}-01",
            "end_date": None if is_current else f"{end_y:04d}-{end_m:02d}-01",
            "duration_months": duration_months,
            "is_current": is_current, "industry": "General",
            "company_size": "201-500", "description": bullets[:600],
        })
        cursor_y, cursor_m = start_y, start_m   # next (older) role ends where this one starts
    return out


def _mock_education(text: str) -> list:
    m = re.search(r"EDUCATION\s*\n(.+?)(?:\n\s*\n|\nSKILLS|\Z)", text, re.DOTALL)
    if not m:
        return []
    line = m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""
    parts = [p.strip() for p in line.split(",")]
    inst = parts[0] if parts else "Unknown"
    degree = parts[1] if len(parts) > 1 else ""
    years = re.search(r"(\d{4})-(\d{4})", line)
    return [{
        "institution": inst, "degree": degree, "field_of_study": "",
        "start_year": int(years.group(1)) if years else 2015,
        "end_year": int(years.group(2)) if years else 2019, "grade": None,
    }]


def _mock_skills(text: str) -> list:
    m = re.search(r"SKILLS\s*\n(.+?)(?:\n\s*\n|\nNotice period|\Z)", text, re.DOTALL)
    if not m:
        return []
    names = [s.strip() for s in m.group(1).replace("\n", " ").split(",") if s.strip()]
    return [{"name": n, "proficiency": "intermediate", "endorsements": 0,
            "duration_months": None} for n in names]


def _mock_projects_summary(text: str) -> str:
    m = re.search(r"PROJECTS\s*\n(.+?)(?:\n\s*\n|\nEDUCATION|\Z)", text, re.DOTALL)
    if not m:
        return ""
    bullets = " ".join(ln.strip("- ").strip() for ln in m.group(1).splitlines() if ln.strip())
    return bullets[:400]


def _mock_profile(text: str) -> dict:
    m = re.search(r"SUMMARY\s*\n(.+?)(?:\n\s*\n|\nEXPERIENCE|\Z)", text, re.DOTALL)
    summary = m.group(1).strip() if m else ""
    yrs_m = re.search(r"([\d.]+)\s*years?", summary)
    loc_m = re.search(r"\|\s*([A-Za-z ]+),\s*India", text)
    first_role = _EXP_ENTRY_RE.search(text)
    return {
        "headline": (first_role.group("title").strip() if first_role else "Professional"),
        "summary": summary[:400],
        "location": loc_m.group(1).strip() if loc_m else "Unknown",
        "country": "India",
        "years_of_experience": float(yrs_m.group(1)) if yrs_m else 0.0,
        "current_title": first_role.group("title").strip() if first_role else "",
        "current_company": first_role.group("company").strip() if first_role else "",
        "current_company_size": "201-500",
        "current_industry": "General",
    }


# --------------------------------------------------------------------------- #
# deterministic mock backend for --mock / offline testing
# --------------------------------------------------------------------------- #
class MockCompareBackend:
    """Deterministic offline backend: exposes both `generate()` (for B/C and
    the resume-compile leaves inside method A) and `embed()` (for method A's
    `score_candidate`). Tier-correlated: scores/embeddings are seeded off the
    resume text so strong/borderline/weak resumes get a real, reproducible
    spread with no network call.

    `generate()` inspects the prompt text for tier-revealing keywords baked
    into the synthetic resumes (see tools/gen_resumes.py) so the mocked LLM
    "judges" fit similarly to a real rubric-following model would, without any
    API call — this keeps the offline `--mock` run exercising the full B/C
    JSON-parsing path with a plausible, non-constant score distribution.
    """

    name = "mock-compare"
    supports_response_schema = False

    def __init__(self, dim: int = 64, seed_salt: str = ""):
        self.dim = dim
        self.seed_salt = seed_salt
        self.last_usage = None
        self.calls = 0

    # -- generate(): used for B/C fit-judging AND the resume-compile leaves -
    def generate(self, prompt: str, system=None, temperature: float = 0.2,
                 max_tokens: int = 2048) -> str:
        self.calls += 1
        self.last_usage = {"prompt_tokens": len(prompt) // 4,
                           "output_tokens": 40, "total_tokens": len(prompt) // 4 + 40}
        # B (rubric) and C (naive) fit-judgment prompts: B asks for the
        # fit_0_100 JSON key explicitly; C's minimal prompt doesn't name a
        # key but is uniquely identifiable by its "rate fit" instruction.
        if "fit_0_100" in prompt or "rate fit" in prompt.lower():
            score = self._tier_score(prompt)
            return json.dumps({
                "fit_0_100": score,
                "reasons": [f"mock-scored via keyword heuristic (score={score})"],
            })
        # resume-compile leaves (harness.resume_prompts): rather than return
        # empty stubs (which would starve method A's career_history-derived
        # dense/evidence channels of any signal and make every tier score
        # ~identically near 0), do a small deterministic regex-based
        # extraction straight off the RESUME: text embedded in the prompt.
        # This is intentionally simple — good enough to give method A a real,
        # tier-differentiated candidate to score, not a claim of NLU quality.
        #
        # IMPORTANT: dispatch on the INSTRUCTION text only (everything before
        # the embedded "RESUME:" block), and on each leaf's UNIQUE requested
        # JSON field name (e.g. "institution" only appears in the education
        # schema) rather than generic words like "education"/"skills" — those
        # generic words appear in more than one leaf's instructions (e.g.
        # career_history_prompt's "not projects, not education" caveat), so
        # matching on them directly would misroute a leaf to the wrong mock.
        resume_text = _extract_resume_block(prompt)
        instruction = prompt.split("RESUME:", 1)[0].lower()
        if '"institution"' in instruction:
            return json.dumps(_mock_education(resume_text))
        if '"company"' in instruction and '"title"' in instruction:
            return json.dumps(_mock_career_history(resume_text))
        if '"proficiency"' in instruction:
            return json.dumps(_mock_skills(resume_text))
        if "projects_summary" in instruction:
            return json.dumps({"projects_summary": _mock_projects_summary(resume_text)})
        if '"headline"' in instruction and '"summary"' in instruction:
            return json.dumps(_mock_profile(resume_text))
        return json.dumps({})

    # -- embed(): used by score_candidate --------------------------------
    def embed(self, texts: list) -> list:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list:
        toks = [t for t in text.lower().split() if t.isalpha()]
        v = np.zeros(self.dim)
        for t in toks:
            h = int(hashlib.sha256((self.seed_salt + t).encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v += rng.normal(size=self.dim)
        if not toks:
            h = int(hashlib.sha256((self.seed_salt + text).encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v = rng.normal(size=self.dim)
        n = float((v ** 2).sum()) ** 0.5
        return (v / n).tolist() if n else v.tolist()

    def _tier_score(self, prompt: str) -> int:
        """Deterministic-ish tier-correlated score: strong resumes mention
        many quantified/domain-matching cues; weak resumes mention few. Uses a
        prompt hash to add a small amount of run-to-run jitter so stability
        metrics have something non-zero to measure for B/C, mirroring a real
        sampled LLM call."""
        low = prompt.lower()
        strong_cues = ["measurable impact", "large-scale", "designed and shipped",
                      "led work", "owned work", "built work"]
        weak_cues = ["unrelated", "administrative", "data entry", "daily coordination"]
        base = 50
        base += 8 * sum(low.count(c) for c in strong_cues)
        base -= 10 * sum(low.count(c) for c in weak_cues)
        base = max(5, min(95, base))
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        jitter = (h % 7) - 3   # +-3 deterministic-per-call-count jitter
        return int(max(0, min(100, base + jitter)))


def make_mock_backend() -> MockCompareBackend:
    return MockCompareBackend()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Comparison study: our engine vs naive single-LLM baselines "
                    "on the labeled synthetic resume set.")
    ap.add_argument("--key", default=None, help="Google AI Studio API key "
                    "(or set GOOGLE_API_KEY)")
    ap.add_argument("--models", default="gemma-4-26b-a4b-it",
                    help="comma-separated model id(s); first one is used")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit to the first N (jd,tier) pairs")
    ap.add_argument("--mock", action="store_true",
                    help="run fully offline with a deterministic mock backend "
                        "(no network, no key)")
    ap.add_argument("--out-dir", default=REPORT_DIR)
    args = ap.parse_args(argv)

    dataset = discover_dataset(limit=args.limit)
    if not dataset:
        print("No (JD, resume) pairs found under data/synthetic_resumes/ — "
              "run tools/gen_resumes.py first.")
        return 2

    if args.mock:
        backend = make_mock_backend()
    else:
        key = args.key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            print("Set --key or GOOGLE_API_KEY to run the live study "
                  "(or pass --mock for an offline dry run).")
            return 2
        from harness.backends import make_backend
        model = args.models.split(",")[0].strip()
        backend = make_backend(key, model)
        if not hasattr(backend, "embed"):
            print(f"Backend for model {model!r} has no embed() — "
                  "method A needs live embeddings.")
            return 2

    def _progress(row: Row) -> None:
        status = "ERR" if row.error else "ok"
        print(f"[{status}] {row.jd_slug}/{row.tier} {row.method} rep={row.repeat} "
              f"score={row.score:.1f}")

    study = run_study(dataset, backend, repeats=args.repeats, on_row=_progress)
    report_path, csv_path = write_report(study, out_dir=args.out_dir)
    print(f"\nwrote {report_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
