"""test_fit_parity — pin fit.compute_components against silent drift.

`redrob_ranker/fit.py` is a hand-port of REPO1's per-candidate math
(`features.py` / `rules.py`). This test freezes the DETERMINISTIC part of that
port (everything that does NOT depend on embeddings / BM25 / pool `mm()`) against
`tests/fixtures/fit_parity_golden.json`, so any accidental change to a ported
formula fails here. The golden is regenerated deliberately (with human review)
via `python -m scripts.gen_fit_parity_golden` — see that script + CONTEXT.md's
"Fit parity fixture" entry. A live cross-run against REPO1 is infeasible
(build_features needs precomputed pool artifacts), so this is a frozen golden
with REPO1 line-ref provenance, not an automated cross-check.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# the fixture candidate builders live under tests/fixtures/
sys.path.insert(0, os.path.join(_HERE, "fixtures"))

import fit_parity_candidates as fx          # noqa: E402
import redrob_ranker as rr                  # noqa: E402
from redrob_ranker.fit import compute_components  # noqa: E402

GOLDEN_PATH = os.path.join(_HERE, "fixtures", "fit_parity_golden.json")
TOL = 1e-6


def _load_golden() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _pinned_keys(golden: dict) -> list:
    return golden["_provenance"]["pinned_keys"]


@pytest.fixture(scope="module")
def profile_method():
    return rr.load(fx.JD_PATH, fx.METHOD_PATH)


@pytest.mark.parametrize("fixture_name", list(fx.FIXTURES.keys()))
def test_components_match_golden(fixture_name, profile_method):
    """compute_components reproduces the frozen golden for each fixture."""
    golden = _load_golden()
    assert fixture_name in golden, (
        f"fixture {fixture_name!r} missing from golden — regenerate via "
        "`python -m scripts.gen_fit_parity_golden`")
    profile, method = profile_method
    comp = compute_components(fx.FIXTURES[fixture_name](), profile, method,
                              ref_date=fx.REF_DATE)
    expected = golden[fixture_name]
    for key in _pinned_keys(golden):
        assert key in comp, f"compute_components dropped pinned key {key!r}"
        got, want = float(comp[key]), float(expected[key])
        assert abs(got - want) <= TOL, (
            f"{fixture_name}.{key} drifted: got {got!r}, golden {want!r} "
            f"(Δ={got - want:.3e}). If this change is intentional, review it "
            f"against REPO1 features.py/rules.py and regenerate the golden.")


def test_golden_covers_every_fixture():
    """The committed golden must have an entry for every fixture (no stale set)."""
    golden = _load_golden()
    fixture_names = set(fx.FIXTURES.keys())
    golden_names = set(golden.keys()) - {"_provenance"}
    assert fixture_names == golden_names, (
        f"fixtures {fixture_names} != golden entries {golden_names} — "
        "regenerate the golden")


def test_tiers_are_distinguishable():
    """Sanity: the fixtures actually span behaviour (weak trips gates the
    strong one doesn't) — otherwise the golden pins nothing meaningful."""
    golden = _load_golden()
    # weak candidate is engineered to trip red flags / gates
    assert golden["weak"]["integrity"] < golden["strong"]["integrity"]
    assert golden["weak"]["only_consulting"] == 1.0
    assert golden["weak"]["hopper"] == 1.0
    assert golden["strong"]["evid_coverage"] > golden["weak"]["evid_coverage"]
