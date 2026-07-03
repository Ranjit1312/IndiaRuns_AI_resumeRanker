"""`MemoryStore` — the DEFAULT backend: plain in-process dicts, zero deps, zero files.

Implements BOTH the `RelationalStore` and `VectorStore` Protocols so a single
instance can back a `Workspace` end-to-end (`store.get_store()` does exactly
this). Nothing here touches disk — safe for HF Spaces / any session-only
deployment, and naturally wiped when the process/session ends.

Vector search uses pure-Python cosine similarity (no numpy requirement) and
keeps embeddings partitioned by `(embedding_model, dim)` so different embedding
spaces are never compared against each other, mirroring the isolation guarantee
`LanceVectorStore` provides for the file-backed path.
"""
from __future__ import annotations

import math

from .schema import (
    CandidateRecordRow,
    CorrectionRow,
    EmbeddingRow,
    FitRunRow,
    ProfileRow,
    ResumeRow,
    Workspace as WorkspaceRow,
)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryStore:
    """Implements both `RelationalStore` and `VectorStore` over plain dicts."""

    def __init__(self):
        self._workspaces: dict[str, WorkspaceRow] = {}
        self._profiles: dict[str, ProfileRow] = {}
        self._resumes: dict[str, ResumeRow] = {}
        self._records: dict[str, CandidateRecordRow] = {}
        self._fit_runs: dict[str, FitRunRow] = {}
        self._corrections: dict[str, CorrectionRow] = {}
        # embeddings partitioned by (embedding_model, dim) -> {embedding_id: row}
        self._embeddings: dict[tuple[str, int], dict[str, EmbeddingRow]] = {}

    # ---- workspaces ------------------------------------------------------ #
    def save_workspace(self, row: WorkspaceRow) -> None:
        self._workspaces[row.workspace_id] = row

    def get_workspace(self, workspace_id: str) -> WorkspaceRow | None:
        return self._workspaces.get(workspace_id)

    def list_workspaces(self) -> list[WorkspaceRow]:
        return sorted(self._workspaces.values(), key=lambda r: r.created_at)

    # ---- profiles -------------------------------------------------------- #
    def save_profile(self, row: ProfileRow) -> None:
        self._profiles[row.profile_id] = row

    def get_profile(self, profile_id: str) -> ProfileRow | None:
        return self._profiles.get(profile_id)

    def list_profiles(self, workspace_id: str) -> list[ProfileRow]:
        rows = [r for r in self._profiles.values() if r.workspace_id == workspace_id]
        return sorted(rows, key=lambda r: r.created_at)

    # ---- resumes ----------------------------------------------------------- #
    def save_resume(self, row: ResumeRow) -> None:
        self._resumes[row.resume_id] = row

    def get_resume(self, resume_id: str) -> ResumeRow | None:
        return self._resumes.get(resume_id)

    def list_resumes(self, workspace_id: str) -> list[ResumeRow]:
        rows = [r for r in self._resumes.values() if r.workspace_id == workspace_id]
        return sorted(rows, key=lambda r: r.created_at)

    # ---- candidate records -------------------------------------------------- #
    def save_candidate_record(self, row: CandidateRecordRow) -> None:
        self._records[row.record_id] = row

    def get_candidate_record(self, record_id: str) -> CandidateRecordRow | None:
        return self._records.get(record_id)

    def list_candidate_records(self, workspace_id: str) -> list[CandidateRecordRow]:
        rows = [r for r in self._records.values() if r.workspace_id == workspace_id]
        return sorted(rows, key=lambda r: r.created_at)

    # ---- fit runs ---------------------------------------------------------- #
    def save_fit_run(self, row: FitRunRow) -> None:
        self._fit_runs[row.run_id] = row

    def list_fit_runs(self, workspace_id: str) -> list[FitRunRow]:
        rows = [r for r in self._fit_runs.values() if r.workspace_id == workspace_id]
        return sorted(rows, key=lambda r: r.created_at)

    # ---- corrections --------------------------------------------------------- #
    def save_correction(self, row: CorrectionRow) -> None:
        self._corrections[row.correction_id] = row

    def list_corrections(self, workspace_id: str) -> list[CorrectionRow]:
        rows = [r for r in self._corrections.values() if r.workspace_id == workspace_id]
        return sorted(rows, key=lambda r: r.created_at)

    # ---- embeddings (vector) ------------------------------------------------ #
    def upsert_embedding(self, row: EmbeddingRow) -> None:
        if len(row.vector) != row.dim:
            raise ValueError(
                f"embedding vector length {len(row.vector)} != declared dim {row.dim}")
        key = (row.embedding_model, row.dim)
        bucket = self._embeddings.setdefault(key, {})
        bucket[row.embedding_id] = row

    def search_embeddings(
        self,
        query_vector: list[float],
        *,
        embedding_model: str,
        dim: int,
        workspace_id: str | None = None,
        owner_type: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[EmbeddingRow, float]]:
        bucket = self._embeddings.get((embedding_model, dim), {})
        scored: list[tuple[EmbeddingRow, float]] = []
        for row in bucket.values():
            if workspace_id is not None and row.workspace_id != workspace_id:
                continue
            if owner_type is not None and row.owner_type != owner_type:
                continue
            scored.append((row, _cosine(query_vector, row.vector)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
