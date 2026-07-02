"""Structured post-hoc logger for the RLM harness.

Records every LLM leaf call with:
  - prompt sent to the model (truncated for display)
  - raw model response text
  - parsed JSON result (or None on failure)
  - elapsed wall-clock time in seconds
  - success / error status

Usage
-----
    log = HarnessLogger()
    result = compile_jd(jd_text, backend, logger=log)
    for entry in log.entries:
        print(entry)          # each entry is a plain dict
    log.write_jsonl("harness_run.log")   # optional on-disk dump
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LeafEntry:
    """One recorded leaf call."""
    leaf: str                          # logical name, e.g. "role", "signal_detail[python]"
    model: str                         # backend.name
    prompt: str                        # full prompt sent
    system: str | None                 # system instruction (may be None for Gemma)
    raw_response: str | None           # exactly what the model returned (None on error)
    parsed: Any                        # extract_json result, or None
    elapsed_s: float                   # wall-clock seconds for this call
    ok: bool                           # True if parsed is not None
    error: str | None = None           # exception message if ok=False
    tokens: dict | None = None         # {"prompt_tokens","output_tokens","total_tokens"} or None

    def as_dict(self) -> dict:
        return {
            "leaf": self.leaf,
            "model": self.model,
            "elapsed_s": round(self.elapsed_s, 3),
            "ok": self.ok,
            "error": self.error,
            "tokens": self.tokens,
            "prompt_preview": self.prompt[:400] + ("…" if len(self.prompt) > 400 else ""),
            "prompt_full": self.prompt,
            "system": self.system,
            "raw_response": self.raw_response,
            "parsed": self.parsed,
        }


class HarnessLogger:
    """Accumulates LeafEntry records produced during a compile_jd() run."""

    def __init__(self) -> None:
        self._entries: list[LeafEntry] = []
        self._run_start: float = time.monotonic()

    # ------------------------------------------------------------------ #
    # recording API (called by llm_query)
    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        leaf: str,
        model: str,
        prompt: str,
        system: str | None,
        raw_response: str | None,
        parsed: Any,
        elapsed_s: float,
        ok: bool,
        error: str | None = None,
        tokens: dict | None = None,
    ) -> None:
        self._entries.append(LeafEntry(
            leaf=leaf, model=model, prompt=prompt, system=system,
            raw_response=raw_response, parsed=parsed,
            elapsed_s=elapsed_s, ok=ok, error=error, tokens=tokens,
        ))

    # ------------------------------------------------------------------ #
    # read API
    # ------------------------------------------------------------------ #
    @property
    def entries(self) -> list[LeafEntry]:
        return list(self._entries)

    @property
    def total_elapsed_s(self) -> float:
        return time.monotonic() - self._run_start

    def summary(self) -> dict:
        ok_count = sum(1 for e in self._entries if e.ok)
        fail_count = len(self._entries) - ok_count
        return {
            "total_calls": len(self._entries),
            "ok": ok_count,
            "failed": fail_count,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "slowest_leaf": (
                max(self._entries, key=lambda e: e.elapsed_s).leaf
                if self._entries else None
            ),
        }

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def write_jsonl(self, path: str) -> None:
        """Append all entries as JSON-lines to *path* (creates if absent)."""
        with open(path, "a", encoding="utf-8") as fh:
            for e in self._entries:
                fh.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
