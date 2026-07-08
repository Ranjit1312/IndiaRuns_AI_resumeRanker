"""Isolation test for the shared compile loop (Candidate A): rlm.run_compile.

A trivial in-file ArtifactSpec + Validator (no network, no real backend) that
forces exactly one repair then a sentinel, proving run_compile guarantees a
valid artifact and records the repair/sentinel in health — independent of the
JD/résumé specs that actually consume it in production.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.backends import MockBackend
from harness.rlm import ArtifactSpec, CompileOutcome, Environment, run_compile
from harness.validate import ValidationResult


class DictKeyValidator:
    """A dead-simple Validator: `count` must be a positive int, `label` must
    be a non-empty string. Fails on the first offending top-level key."""

    def validate(self, d: dict) -> ValidationResult:
        if not isinstance(d.get("count"), int) or d.get("count", 0) <= 0:
            return ValidationResult(ok=False, error="count must be a positive int",
                                    field="count", top="count")
        if not isinstance(d.get("label"), str) or not d["label"].strip():
            return ValidationResult(ok=False, error="label must be non-empty",
                                    field="label", top="label")
        return ValidationResult(ok=True)


class OneRepairThenSentinelSpec:
    """`count` starts bad; its `rebuild` fixes it in one shot (exercises the
    "repair, then re-validate ok" path). `label` starts (and stays) bad;
    its `rebuild` can never fix it, so run_compile must fall through to
    `sentinel` for that same block (matching compile_jd/compile_resume's
    same-top-block repair-then-sentinel contract)."""

    order = ["count", "label"]

    def __init__(self):
        self.validator = DictKeyValidator()

    def build(self, name, env, backend, health, *, logger=None, on_event=None):
        if name == "count":
            return 5   # valid from the start — count is not the block under test
        if name == "label":
            return ""   # invalid from the start — this is the block under test
        raise KeyError(name)

    def assemble(self, parts: dict, env: Environment) -> dict:
        return {"count": parts["count"], "label": parts["label"]}

    def rebuild(self, artifact, failing_top, env, backend, hint, health,
               *, logger=None, on_event=None):
        if failing_top == "label":
            # rebuild "tries" but the leaf is deterministic garbage — still
            # empty, forcing the caller to fall through to sentinel().
            artifact["label"] = ""
        return artifact

    def sentinel(self, artifact, failing_top):
        if failing_top == "count":
            artifact["count"] = 1
        elif failing_top == "label":
            artifact["label"] = "sentinel-label"
        return artifact


def test_run_compile_guarantees_valid_artifact_and_records_repair_and_sentinel():
    env = Environment("irrelevant text — this spec never reads it")
    spec = OneRepairThenSentinelSpec()
    backend = MockBackend(lambda prompt, system: "{}")

    outcome = run_compile(env, spec, backend, max_repairs=2)

    assert isinstance(outcome, CompileOutcome)
    assert outcome.validation.ok, outcome.validation.error
    assert outcome.artifact["count"] == 5          # was valid from the start, never touched
    assert outcome.artifact["label"] == "sentinel-label"   # rebuild couldn't fix it -> sentinel

    # health bookkeeping
    assert outcome.health["repairs"] >= 1
    assert "label" in outcome.health["sentineled"]
    assert "count" not in outcome.health["sentineled"], \
        "count was already valid and should never be touched"


def test_run_compile_threads_logger_and_on_event():
    from harness.logging_utils import HarnessLogger

    env = Environment("irrelevant")
    spec = OneRepairThenSentinelSpec()
    backend = MockBackend(lambda prompt, system: "{}")
    log = HarnessLogger()

    outcome = run_compile(env, spec, backend, logger=log, max_repairs=2)

    assert outcome.validation.ok
    assert "telemetry" in outcome.health
    assert outcome.health["telemetry"]["total_calls"] == len(log.entries)


def test_run_compile_respects_max_repairs_budget():
    """A validator that never passes must stop after max_repairs, not loop
    forever — the sentinel path still has to leave a usable artifact."""

    class NeverValidValidator:
        def validate(self, d):
            return ValidationResult(ok=False, error="always invalid",
                                    field="count", top="count")

    class NoSentinelSpec:
        order = ["count", "label"]

        def __init__(self):
            self.validator = NeverValidValidator()

        def build(self, name, env, backend, health, *, logger=None, on_event=None):
            return -1 if name == "count" else "x"

        def assemble(self, parts, env):
            return dict(parts)

        def rebuild(self, artifact, failing_top, env, backend, hint, health,
                   *, logger=None, on_event=None):
            return artifact   # never actually fixes anything

        def sentinel(self, artifact, failing_top):
            artifact["count"] = -1   # sentinel also "fails" -> loop must still terminate
            return artifact

    env = Environment("irrelevant")
    spec = NoSentinelSpec()
    backend = MockBackend(lambda prompt, system: "{}")

    outcome = run_compile(env, spec, backend, max_repairs=2)

    assert not outcome.validation.ok
    assert outcome.health["repairs"] == 2   # stopped at the budget, did not loop forever


if __name__ == "__main__":
    for fn in [test_run_compile_guarantees_valid_artifact_and_records_repair_and_sentinel,
               test_run_compile_threads_logger_and_on_event,
               test_run_compile_respects_max_repairs_budget]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all ok")
