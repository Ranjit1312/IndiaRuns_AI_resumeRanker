"""`DuckDBStore` — `RelationalStore` CRUD implementation backed by a DuckDB file.

Optional, file-backed persistence. `duckdb` is imported lazily inside `__init__`
so importing this module (or `store/` as a whole) never requires the package to
be installed unless this backend is actually selected (`STORE_BACKEND=duckdb+lance`).
JSON-shaped columns (candidate_json, result_json, ...) are stored as JSON text and
(de)serialized here, keeping `schema.py` free of any serialization concerns.
"""
from __future__ import annotations

import json

from .bootstrap import create_duckdb
from .schema import (
    CandidateRecordRow,
    CorrectionRow,
    FitRunRow,
    ProfileRow,
    ResumeRow,
    Workspace as WorkspaceRow,
)


class DuckDBStore:
    """Implements the `RelationalStore` Protocol against a DuckDB file."""

    def __init__(self, path: str):
        self._path = path
        self._conn = create_duckdb(path)   # also ensures schema (idempotent)

    # ---- workspaces ---------------------------------------------------- #
    def save_workspace(self, row: WorkspaceRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO workspaces VALUES (?, ?, ?, ?)",
            [row.workspace_id, row.name, row.created_at, row.updated_at])

    def get_workspace(self, workspace_id: str) -> WorkspaceRow | None:
        r = self._conn.execute(
            "SELECT workspace_id, name, created_at, updated_at FROM workspaces "
            "WHERE workspace_id = ?", [workspace_id]).fetchone()
        return WorkspaceRow(*r) if r else None

    def list_workspaces(self) -> list[WorkspaceRow]:
        rows = self._conn.execute(
            "SELECT workspace_id, name, created_at, updated_at FROM workspaces "
            "ORDER BY created_at").fetchall()
        return [WorkspaceRow(*r) for r in rows]

    # ---- profiles -------------------------------------------------------- #
    def save_profile(self, row: ProfileRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [row.profile_id, row.workspace_id, row.name, row.profile_yaml,
             row.meta_yaml, json.dumps(row.source), row.created_at, row.updated_at])

    def get_profile(self, profile_id: str) -> ProfileRow | None:
        r = self._conn.execute(
            "SELECT profile_id, workspace_id, name, profile_yaml, meta_yaml, "
            "source_json, created_at, updated_at FROM profiles WHERE profile_id = ?",
            [profile_id]).fetchone()
        return _row_to_profile(r) if r else None

    def list_profiles(self, workspace_id: str) -> list[ProfileRow]:
        rows = self._conn.execute(
            "SELECT profile_id, workspace_id, name, profile_yaml, meta_yaml, "
            "source_json, created_at, updated_at FROM profiles WHERE workspace_id = ? "
            "ORDER BY created_at", [workspace_id]).fetchall()
        return [_row_to_profile(r) for r in rows]

    # ---- resumes ------------------------------------------------------- #
    def save_resume(self, row: ResumeRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO resumes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [row.resume_id, row.workspace_id, row.name, row.raw_text,
             json.dumps(row.candidate_json), row.created_at, row.updated_at])

    def get_resume(self, resume_id: str) -> ResumeRow | None:
        r = self._conn.execute(
            "SELECT resume_id, workspace_id, name, raw_text, candidate_json, "
            "created_at, updated_at FROM resumes WHERE resume_id = ?",
            [resume_id]).fetchone()
        return _row_to_resume(r) if r else None

    def list_resumes(self, workspace_id: str) -> list[ResumeRow]:
        rows = self._conn.execute(
            "SELECT resume_id, workspace_id, name, raw_text, candidate_json, "
            "created_at, updated_at FROM resumes WHERE workspace_id = ? "
            "ORDER BY created_at", [workspace_id]).fetchall()
        return [_row_to_resume(r) for r in rows]

    # ---- candidate records ----------------------------------------------- #
    def save_candidate_record(self, row: CandidateRecordRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO candidate_records VALUES (?, ?, ?, ?, ?)",
            [row.record_id, row.workspace_id, row.resume_id,
             json.dumps(row.candidate_json), row.created_at])

    def get_candidate_record(self, record_id: str) -> CandidateRecordRow | None:
        r = self._conn.execute(
            "SELECT record_id, workspace_id, resume_id, candidate_json, created_at "
            "FROM candidate_records WHERE record_id = ?", [record_id]).fetchone()
        return _row_to_record(r) if r else None

    def list_candidate_records(self, workspace_id: str) -> list[CandidateRecordRow]:
        rows = self._conn.execute(
            "SELECT record_id, workspace_id, resume_id, candidate_json, created_at "
            "FROM candidate_records WHERE workspace_id = ? ORDER BY created_at",
            [workspace_id]).fetchall()
        return [_row_to_record(r) for r in rows]

    # ---- fit runs -------------------------------------------------------- #
    def save_fit_run(self, row: FitRunRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO fit_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [row.run_id, row.workspace_id, row.profile_id, row.record_id,
             json.dumps(row.result_json), row.overall, row.created_at])

    def list_fit_runs(self, workspace_id: str) -> list[FitRunRow]:
        rows = self._conn.execute(
            "SELECT run_id, workspace_id, profile_id, record_id, result_json, "
            "overall, created_at FROM fit_runs WHERE workspace_id = ? "
            "ORDER BY created_at", [workspace_id]).fetchall()
        return [_row_to_fit_run(r) for r in rows]

    # ---- corrections ------------------------------------------------------ #
    def save_correction(self, row: CorrectionRow) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [row.correction_id, row.workspace_id, row.resume_id, row.field_path,
             json.dumps(row.before), json.dumps(row.after), row.note, row.created_at])

    def list_corrections(self, workspace_id: str) -> list[CorrectionRow]:
        rows = self._conn.execute(
            "SELECT correction_id, workspace_id, resume_id, field_path, "
            "before_json, after_json, note, created_at FROM corrections "
            "WHERE workspace_id = ? ORDER BY created_at", [workspace_id]).fetchall()
        return [_row_to_correction(r) for r in rows]


# --------------------------------------------------------------------------- #
# row (de)serialization helpers
# --------------------------------------------------------------------------- #
def _row_to_profile(r) -> ProfileRow:
    return ProfileRow(profile_id=r[0], workspace_id=r[1], name=r[2],
                       profile_yaml=r[3], meta_yaml=r[4] or "",
                       source=json.loads(r[5]) if r[5] else {},
                       created_at=r[6], updated_at=r[7] or "")


def _row_to_resume(r) -> ResumeRow:
    return ResumeRow(resume_id=r[0], workspace_id=r[1], name=r[2],
                      raw_text=r[3] or "",
                      candidate_json=json.loads(r[4]) if r[4] else {},
                      created_at=r[5], updated_at=r[6] or "")


def _row_to_record(r) -> CandidateRecordRow:
    return CandidateRecordRow(record_id=r[0], workspace_id=r[1], resume_id=r[2],
                               candidate_json=json.loads(r[3]) if r[3] else {},
                               created_at=r[4])


def _row_to_fit_run(r) -> FitRunRow:
    return FitRunRow(run_id=r[0], workspace_id=r[1], profile_id=r[2], record_id=r[3],
                      result_json=json.loads(r[4]) if r[4] else {},
                      overall=r[5] or 0.0, created_at=r[6])


def _row_to_correction(r) -> CorrectionRow:
    return CorrectionRow(correction_id=r[0], workspace_id=r[1], resume_id=r[2],
                          field_path=r[3],
                          before=json.loads(r[4]) if r[4] is not None else None,
                          after=json.loads(r[5]) if r[5] is not None else None,
                          note=r[6] or "", created_at=r[7])
