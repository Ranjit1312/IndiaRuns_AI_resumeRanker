"""The swap seam — `RelationalStore` / `VectorStore` Protocols + the `Workspace` facade.

Callers (app.py, harness/*, redrob_ranker/fit.py) never talk to DuckDB, LanceDB, or
plain dicts directly — they only ever hold a `Workspace`. `Workspace` composes one
`RelationalStore` (required) and one `VectorStore` (optional — a workspace with no
vector backend simply can't do semantic search). Swapping `MemoryStore` for
`DuckDBStore`+`LanceVectorStore`, or later for an enterprise Postgres/pgvector
implementation, means writing a new class that satisfies these Protocols — nothing
else in the codebase changes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schema import (
    CandidateRecordRow,
    CorrectionRow,
    EmbeddingRow,
    FitRunRow,
    ProfileRow,
    ResumeRow,
    Workspace as WorkspaceRow,
)


# --------------------------------------------------------------------------- #
# Protocols — the swap seam
# --------------------------------------------------------------------------- #
@runtime_checkable
class RelationalStore(Protocol):
    """CRUD surface any relational backend (MemoryStore, DuckDBStore, an
    enterprise Postgres impl, ...) must implement."""

    # workspaces
    def save_workspace(self, row: WorkspaceRow) -> None: ...
    def get_workspace(self, workspace_id: str) -> WorkspaceRow | None: ...
    def list_workspaces(self) -> list[WorkspaceRow]: ...

    # profiles
    def save_profile(self, row: ProfileRow) -> None: ...
    def get_profile(self, profile_id: str) -> ProfileRow | None: ...
    def list_profiles(self, workspace_id: str) -> list[ProfileRow]: ...

    # resumes
    def save_resume(self, row: ResumeRow) -> None: ...
    def get_resume(self, resume_id: str) -> ResumeRow | None: ...
    def list_resumes(self, workspace_id: str) -> list[ResumeRow]: ...

    # candidate records
    def save_candidate_record(self, row: CandidateRecordRow) -> None: ...
    def get_candidate_record(self, record_id: str) -> CandidateRecordRow | None: ...
    def list_candidate_records(self, workspace_id: str) -> list[CandidateRecordRow]: ...

    # fit runs
    def save_fit_run(self, row: FitRunRow) -> None: ...
    def list_fit_runs(self, workspace_id: str) -> list[FitRunRow]: ...

    # corrections
    def save_correction(self, row: CorrectionRow) -> None: ...
    def list_corrections(self, workspace_id: str) -> list[CorrectionRow]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector upsert/search surface. Every implementation MUST keep vectors from
    different (embedding_model, dim) pairs isolated from one another — never
    compare cosine similarity across embedding models."""

    def upsert_embedding(self, row: EmbeddingRow) -> None: ...

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
        """Return up to `top_k` (row, cosine_similarity) pairs, best first,
        restricted to the given (embedding_model, dim)."""
        ...


# --------------------------------------------------------------------------- #
# Workspace facade
# --------------------------------------------------------------------------- #
class Workspace:
    """Unifies a `RelationalStore` + optional `VectorStore` behind one API.

    This is the only object application code should hold. `relational` is
    required; `vector` is optional (a session with no vector backend can still
    do everything except `upsert_embedding`/`search_embeddings`).
    """

    def __init__(self, relational: RelationalStore, vector: VectorStore | None = None):
        self.relational = relational
        self.vector = vector

    # ---- workspaces -------------------------------------------------- #
    def save_workspace(self, row: WorkspaceRow) -> None:
        self.relational.save_workspace(row)

    def get_workspace(self, workspace_id: str) -> WorkspaceRow | None:
        return self.relational.get_workspace(workspace_id)

    def list_workspaces(self) -> list[WorkspaceRow]:
        return self.relational.list_workspaces()

    # ---- profiles ------------------------------------------------------ #
    def save_profile(self, row: ProfileRow) -> None:
        self.relational.save_profile(row)

    def get_profile(self, profile_id: str) -> ProfileRow | None:
        return self.relational.get_profile(profile_id)

    def list_profiles(self, workspace_id: str) -> list[ProfileRow]:
        return self.relational.list_profiles(workspace_id)

    # ---- resumes --------------------------------------------------------- #
    def save_resume(self, row: ResumeRow) -> None:
        self.relational.save_resume(row)

    def get_resume(self, resume_id: str) -> ResumeRow | None:
        return self.relational.get_resume(resume_id)

    def list_resumes(self, workspace_id: str) -> list[ResumeRow]:
        return self.relational.list_resumes(workspace_id)

    # ---- candidate records ------------------------------------------------ #
    def save_candidate_record(self, row: CandidateRecordRow) -> None:
        self.relational.save_candidate_record(row)

    def get_candidate_record(self, record_id: str) -> CandidateRecordRow | None:
        return self.relational.get_candidate_record(record_id)

    def list_candidate_records(self, workspace_id: str) -> list[CandidateRecordRow]:
        return self.relational.list_candidate_records(workspace_id)

    # ---- fit runs ----------------------------------------------------- #
    def save_fit_run(self, row: FitRunRow) -> None:
        self.relational.save_fit_run(row)

    def list_fit_runs(self, workspace_id: str) -> list[FitRunRow]:
        return self.relational.list_fit_runs(workspace_id)

    # ---- corrections ---------------------------------------------------- #
    def save_correction(self, row: CorrectionRow) -> None:
        self.relational.save_correction(row)

    def list_corrections(self, workspace_id: str) -> list[CorrectionRow]:
        return self.relational.list_corrections(workspace_id)

    # ---- embeddings (vector, optional) --------------------------------- #
    def upsert_embedding(self, row: EmbeddingRow) -> None:
        if self.vector is None:
            raise RuntimeError("Workspace has no VectorStore configured")
        self.vector.upsert_embedding(row)

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
        if self.vector is None:
            raise RuntimeError("Workspace has no VectorStore configured")
        return self.vector.search_embeddings(
            query_vector, embedding_model=embedding_model, dim=dim,
            workspace_id=workspace_id, owner_type=owner_type, top_k=top_k)
