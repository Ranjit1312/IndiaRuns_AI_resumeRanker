"""gen_fit_parity_golden.py — regenerate tests/fixtures/fit_parity_golden.json.

Run this after a DELIBERATE change to redrob_ranker.fit.compute_components
(e.g. a real formula fix ported from REPO1). It recomputes the golden from
the CURRENT code and overwrites the committed JSON.

IMPORTANT — human review required before committing the result:
    Before committing the regenerated golden, eyeball the diff (`git diff
    tests/fixtures/fit_parity_golden.json`) against REPO1's
    `redrob_ranker/features.py` and `redrob_ranker/rules.py` (line ranges
    cited in fit.py's module docstring) to confirm the new numbers still
    match a faithful port of those formulas. This script does NOT verify
    correctness — it only snapshots whatever compute_components currently
    returns. tests/test_fit_parity.py is what catches accidental drift; this
    script is for INTENTIONAL, reviewed changes only.

If the sibling Repo 1 checkout is present at the expected relative path,
this script ALSO runs a best-effort additional check: it imports Repo 1's
`features.py`/`rules.py` and reports whether they're importable (informational
only — Repo 1's `build_features` needs precomputed pool artifacts, so it
cannot cheaply score a single fixture candidate; see CONTEXT.md's "Fit parity
fixture" entry). This script never fails just because Repo 1 is absent.

Usage:
    ./.venv/Scripts/python -m scripts.gen_fit_parity_golden
"""
from __future__ import annotations

import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "fixtures"))

import redrob_ranker as rr
from redrob_ranker.fit import compute_components

import fit_parity_candidates as fx

GOLDEN_PATH = os.path.join(_ROOT, "tests", "fixtures", "fit_parity_golden.json")

# REPO1 line-ranges these deterministic formulas port (from fit.py's module
# docstring) — kept here too so the golden is self-documenting provenance.
_PROVENANCE_NOTE = {
    "source": "REPO1 redrob_ranker/features.py + redrob_ranker/rules.py "
              "(sibling repo, read-only reference)",
    "repo1_line_refs": {
        "intrinsic_row": "intrinsic.extract_intrinsic([candidate]).iloc[0]",
        "dense_per_signal_and_pooling": "features.py L217-236",
        "evidence_coverage_and_depth_bonus": "features.py L238-264",
        "cv_primary_domain_nlp_ratio_ai_corroboration": "features.py L266-305",
        "yoe_fit_gaussian": "features.py L323",
        "hopper": "features.py L327",
        "only_consulting_and_months_since_ic_role": "features.py L183-215",
        "location_ladder_loc2": "features.py L330-363",
        "integrity_ladder": "features.py L365-394",
        "availability": "features.py L397-415",
        "notice_pen": "features.py L418-423",
        "assess_strength": "features.py L308-318",
        "composite_structure_without_mm": "rules.py L119-178",
    },
    "jd_profile_used": "data/eval_jds/stripe_backend-software-engineer/jd_profile.yaml",
    "method_config_used": "jd/method_config.yaml",
    "ref_date": fx.REF_DATE,
    "pinned_keys": [
        "yoe_fit", "hopper", "only_consulting", "months_since_ic_role",
        "cv_primary", "domain_nlp_ratio", "ai_skill_corroboration",
        "ai_skills_claimed", "evid_coverage", "depth_bonus", "assess_strength",
        "integrity", "availability_mult", "notice_pen", "loc2_v4",
    ],
    "generated_on": None,   # filled in at write time
    "note": "This is a FROZEN GOLDEN of compute_components' deterministic "
           "output, not a live cross-check against REPO1 (REPO1's "
           "build_features needs precomputed pool artifacts — "
           "job_embeddings.npy, bm25_facets.parquet, mm() over a pool — so "
           "it cannot cheaply score a single fixture candidate). Regenerate "
           "deliberately via this script and review the diff by hand "
           "against REPO1 features.py/rules.py before committing.",
}

_PINNED_KEYS = _PROVENANCE_NOTE["pinned_keys"]


def _check_repo1_sibling_present() -> "str | None":
    """Best-effort: report whether the sibling Repo 1 checkout is importable.

    Returns a short status string for the console; never raises. This is
    informational only per CONTEXT.md's Fit parity fixture note — Repo 1's
    build_features cannot run without precomputed pool artifacts, so this
    does NOT attempt to cross-run it, only checks presence/importability.
    """
    candidate_root = os.path.normpath(os.path.join(
        _ROOT, "..", "India_runs_data_and_ai_challenge"))
    features_path = os.path.join(candidate_root, "redrob_ranker", "features.py")
    rules_path = os.path.join(candidate_root, "redrob_ranker", "rules.py")
    if not (os.path.isfile(features_path) and os.path.isfile(rules_path)):
        return f"sibling Repo 1 not found at {candidate_root!r} (skipping)"
    return (f"sibling Repo 1 found at {candidate_root!r} "
           f"(features.py, rules.py present) — live cross-check still "
           f"infeasible: build_features needs precomputed pool artifacts")


def generate() -> dict:
    profile, method = rr.load(fx.JD_PATH, fx.METHOD_PATH)
    golden = {"_provenance": dict(_PROVENANCE_NOTE)}
    golden["_provenance"]["generated_on"] = datetime.date.today().isoformat()

    for name, builder in fx.FIXTURES.items():
        comp = compute_components(builder(), profile, method, ref_date=fx.REF_DATE)
        golden[name] = {k: comp[k] for k in _PINNED_KEYS}
    return golden


def main() -> int:
    status = _check_repo1_sibling_present()
    print(f"[gen_fit_parity_golden] {status}")

    golden = generate()
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[gen_fit_parity_golden] wrote {GOLDEN_PATH}")
    print("[gen_fit_parity_golden] REVIEW THE DIFF against REPO1 "
         "features.py/rules.py before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
