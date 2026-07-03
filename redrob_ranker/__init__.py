"""redrob_ranker — v7 JD-seam pure core (Phase 2 subset).

Public surface:
    load(jd_path, method_path) -> (Profile, Method)   [profile.py]
    Profile, Method, Signal, RoleSpec, LocSpec, DomainSpec
    extract_intrinsic(records) -> DataFrame, INTRINSIC_COLUMNS  [intrinsic.py]
    score_candidate(candidate, profile, method, backend) -> FitResult  [fit.py]

(rules.py / features.py / gates.py are NOT ported wholesale — they are pool/
artifact-based. fit.py ports their per-candidate math faithfully; see its
module docstring and docs/PHASE2_SPEC.md Part C1.)
"""
from .profile import (load, Profile, Method, Signal, RoleSpec, LocSpec,
                      DomainSpec, RedFlag)
from .intrinsic import extract_intrinsic, INTRINSIC_COLUMNS
from .fit import score_candidate, FitResult, SignalScore, GateResult

__all__ = [
    "load", "Profile", "Method", "Signal", "RoleSpec", "LocSpec",
    "DomainSpec", "RedFlag",
    "extract_intrinsic", "INTRINSIC_COLUMNS",
    "score_candidate", "FitResult", "SignalScore", "GateResult",
]
