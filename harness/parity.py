"""Multi-model parity eval — does the harness coerce different models to the same
schema-valid, semantically-consistent profile?

Two modes:
  * `run_parity(jd, models)`  — compile one JD across models; report per-model
    validity/health + cross-model structural agreement.
  * `run_vs_gold(dir, models)` — compile each eval JD (`data/eval_jds/<slug>/jd.txt`)
    and compare structural fields to that slug's hand-authored gold jd_profile.yaml.

Structural agreement = exact/near-exact on the fields that must be stable
(red_flags, notice, YoE band, signal count, domain-term overlap). Free-text
fields (query/regex/cross_encoder) vary by wording and are NOT diffed here; a
gemini-embedding cosine is the intended upgrade (TODO — needs the same key).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import yaml

from .backends import DEFAULT_MODEL, make_backend
from .coerce import compile_jd
from .validate import validate_profile_dict


def _structural(profile: dict) -> dict:
    r = profile.get("role", {})
    ie = r.get("ideal_experience", {})
    return {
        "n_signals": len(profile.get("signals", [])),
        "signal_ids": sorted(s["id"] for s in profile.get("signals", [])),
        "notice": r.get("notice_preference_days"),
        "peak_years": ie.get("peak_years"),
        "red_flags_on": sorted(k for k, v in profile.get("red_flags", {}).items()
                               if v.get("enabled")),
        "out_domain": sorted(profile.get("domain", {}).get("out_of_domain_terms", [])),
    }


def _overlap(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def run_parity(jd_text: str, models: list[str], key: str, max_repairs: int = 2) -> dict:
    rows = {}
    for m in models:
        res = compile_jd(jd_text, make_backend(key, m), max_repairs=max_repairs)
        rows[m] = {"valid": res.validation.ok, "health": res.health,
                   "structural": _structural(res.profile)}
    # cross-model agreement vs the first model
    base = next(iter(rows))
    bs = rows[base]["structural"]
    for m, row in rows.items():
        s = row["structural"]
        row["agreement_vs_base"] = {
            "same_red_flags": s["red_flags_on"] == bs["red_flags_on"],
            "signal_count_delta": s["n_signals"] - bs["n_signals"],
            "out_domain_overlap": round(_overlap(s["out_domain"], bs["out_domain"]), 2),
        }
    return {"base_model": base, "models": rows}


def run_vs_gold(eval_dir: str, models: list[str], key: str) -> dict:
    out = {}
    for jd_path in sorted(glob.glob(os.path.join(eval_dir, "*", "jd.txt"))):
        slug = os.path.basename(os.path.dirname(jd_path))
        gold_path = os.path.join(os.path.dirname(jd_path), "jd_profile.yaml")
        if not os.path.exists(gold_path):
            continue
        gold = yaml.safe_load(open(gold_path, encoding="utf-8"))
        gs = _structural(gold)
        jd = open(jd_path, encoding="utf-8").read()
        per_model = {}
        for m in models:
            res = compile_jd(jd, make_backend(key, m), max_repairs=2)
            s = _structural(res.profile)
            per_model[m] = {
                "valid": res.validation.ok,
                "same_red_flags": s["red_flags_on"] == gs["red_flags_on"],
                "signal_count": (s["n_signals"], gs["n_signals"]),
                "out_domain_overlap": round(_overlap(s["out_domain"], gs["out_domain"]), 2),
            }
        out[slug] = per_model
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Multi-model parity eval for the JD harness")
    ap.add_argument("--jd", help="compile one JD file across --models")
    ap.add_argument("--eval-dir", help="run each data/eval_jds/*/jd.txt vs its gold")
    ap.add_argument("--models", default=f"{DEFAULT_MODEL},gemma-4-31b-it")
    args = ap.parse_args(argv)
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Set GOOGLE_API_KEY to run parity.")
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.jd:
        report = run_parity(open(args.jd, encoding="utf-8").read(), models, key)
    elif args.eval_dir:
        report = run_vs_gold(args.eval_dir, models, key)
    else:
        print("Pass --jd <file> or --eval-dir data/eval_jds")
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
