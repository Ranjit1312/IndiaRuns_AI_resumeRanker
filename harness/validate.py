"""Authoritative validation — reuse the engine's own validator.

We never re-implement the JD schema. `redrob_ranker.profile.load` is the single
source of truth: it runs JSON-Schema validation, compiles every regex, and
raises `ValueError("jd_profile.yaml: field '<path>' <msg>")`. We parse that
field path so the RLM harness can re-call just the failing leaf.

The retained `redrob_ranker/__init__.py` imports nothing heavy, so this stays
numpy/pandas-free (Module 1 is light).
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import yaml

from redrob_ranker import profile as rprofile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
METHOD_PATH = os.path.join(_ROOT, "jd", "method_config.yaml")

# "jd_profile.yaml: field 'signals.2.evidence_regex' ..."  or "field 'role.title'"
_FIELD_RE = re.compile(r"field '([^']+)'")


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None          # full precise message
    field: str | None = None          # offending dotted path, if parseable
    top: str | None = None            # top-level block (e.g. "signals", "role")


@runtime_checkable
class Validator(Protocol):
    """The seam `(dict) -> ValidationResult` (Candidate C).

    An `ArtifactSpec` injects one of these; `run_compile` never knows which
    mechanism actually runs (engine profile.load vs jsonschema Draft-7, ...).
    """

    def validate(self, d: dict) -> ValidationResult:
        ...


def field_of(error: str | None) -> tuple[str | None, str | None]:
    if not error:
        return None, None
    m = _FIELD_RE.search(error)
    if not m:
        # regex-compile errors read "invalid regex in field 'signals[0]...'"
        m = re.search(r"field '([^']+)'", error)
    if not m:
        return None, None
    path = m.group(1)
    top = re.split(r"[.\[]", path, 1)[0]
    return path, top


class EngineProfileValidator:
    """Adapter: validates a jd_profile dict via the engine's own `profile.load`.

    Writes the dict to a temp YAML file (the engine loader is file-based),
    invokes `redrob_ranker.profile.load`, and translates the resulting
    `ValueError`'s field path into a `ValidationResult`. Pure/no mutation.
    """

    def __init__(self, method_path: str = METHOD_PATH) -> None:
        self.method_path = method_path

    def validate(self, d: dict) -> ValidationResult:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(d, tmp, sort_keys=False, allow_unicode=True)
            tmp.close()
            rprofile.load(tmp.name, self.method_path)   # raises ValueError on any problem
            return ValidationResult(ok=True)
        except ValueError as exc:
            field, top = field_of(str(exc))
            return ValidationResult(ok=False, error=str(exc), field=field, top=top)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def validate_profile_dict(prof: dict, method_path: str = METHOD_PATH) -> ValidationResult:
    """Validate a jd_profile dict via the engine's profile.load. Pure/no mutation.

    Thin back-compat wrapper — delegates to `EngineProfileValidator`."""
    return EngineProfileValidator(method_path).validate(prof)
