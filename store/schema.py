"""Schema definitions ONLY — row dataclasses + DDL strings.

Pure data: no connections, no `import duckdb`/`import lancedb`, no I/O. This module
is the single source of truth for the shape of every table in the store/ layer.
`store/bootstrap.py` reads `TABLE_DDL` to create tables; `store/*_store.py` impls
read the dataclasses to know what a row looks like. Keeping this file inert means
swapping the backing engine (DuckDB → an enterprise Postgres, say) never touches
"what a row is" — only "how it's persisted".

Tables: workspaces, profiles, resumes, candidate_records, fit_runs, corrections,
embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# row models
# --------------------------------------------------------------------------- #
@dataclass
class Workspace:
    workspace_id: str
    name: str
    created_at: str            # ISO-8601 timestamp
    updated_at: str = ""


@dataclass
class ProfileRow:
    profile_id: str
    workspace_id: str
    name: str                  # human label, e.g. "Senior Data Engineer — Acme"
    profile_yaml: str          # jd_profile.yaml text
    meta_yaml: str = ""        # jd_meta.yaml text (coaching sidecar), optional
    source: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ResumeRow:
    resume_id: str
    workspace_id: str
    name: str                  # human label, e.g. candidate display name
    raw_text: str               # extracted resume text
    candidate_json: dict = field(default_factory=dict)   # compiled candidate dict
    created_at: str = ""
    updated_at: str = ""


@dataclass
class CandidateRecordRow:
    """A validated candidate dict (candidate_schema.json), snapshotted at a point
    in time so fit_runs can reference an immutable input even if the resume/
    profile is later edited."""
    record_id: str
    workspace_id: str
    resume_id: str
    candidate_json: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class FitRunRow:
    run_id: str
    workspace_id: str
    profile_id: str
    record_id: str
    result_json: dict = field(default_factory=dict)   # FitResult.to_dict()
    overall: float = 0.0
    created_at: str = ""


@dataclass
class CorrectionRow:
    """A parsed-vs-approved delta captured during HITL review."""
    correction_id: str
    workspace_id: str
    resume_id: str
    field_path: str             # dotted path into the candidate/profile dict
    before: object = None
    after: object = None
    note: str = ""
    created_at: str = ""


@dataclass
class EmbeddingRow:
    """One vector row. `embedding_model` + `dim` MUST both be carried so that
    different embedding models (e.g. EmbeddingGemma vs Gemini) never get mixed
    inside the same vector table."""
    embedding_id: str
    workspace_id: str
    owner_type: str             # e.g. "resume", "profile_signal", "job_chunk"
    owner_id: str
    embedding_model: str
    dim: int
    vector: list = field(default_factory=list)
    text: str = ""              # the text that was embedded (for debugging/audit)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""


# --------------------------------------------------------------------------- #
# DDL — relational tables (DuckDB-flavored SQL; used only by bootstrap.py)
# --------------------------------------------------------------------------- #
TABLE_DDL: dict[str, str] = {
    "workspaces": """
        CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id VARCHAR PRIMARY KEY,
            name         VARCHAR NOT NULL,
            created_at   VARCHAR NOT NULL,
            updated_at   VARCHAR
        )
    """,
    "profiles": """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id   VARCHAR PRIMARY KEY,
            workspace_id VARCHAR NOT NULL,
            name         VARCHAR NOT NULL,
            profile_yaml VARCHAR NOT NULL,
            meta_yaml    VARCHAR,
            source_json  VARCHAR,
            created_at   VARCHAR NOT NULL,
            updated_at   VARCHAR
        )
    """,
    "resumes": """
        CREATE TABLE IF NOT EXISTS resumes (
            resume_id      VARCHAR PRIMARY KEY,
            workspace_id   VARCHAR NOT NULL,
            name           VARCHAR NOT NULL,
            raw_text       VARCHAR,
            candidate_json VARCHAR,
            created_at     VARCHAR NOT NULL,
            updated_at     VARCHAR
        )
    """,
    "candidate_records": """
        CREATE TABLE IF NOT EXISTS candidate_records (
            record_id      VARCHAR PRIMARY KEY,
            workspace_id   VARCHAR NOT NULL,
            resume_id      VARCHAR NOT NULL,
            candidate_json VARCHAR,
            created_at     VARCHAR NOT NULL
        )
    """,
    "fit_runs": """
        CREATE TABLE IF NOT EXISTS fit_runs (
            run_id       VARCHAR PRIMARY KEY,
            workspace_id VARCHAR NOT NULL,
            profile_id   VARCHAR NOT NULL,
            record_id    VARCHAR NOT NULL,
            result_json  VARCHAR,
            overall      DOUBLE,
            created_at   VARCHAR NOT NULL
        )
    """,
    "corrections": """
        CREATE TABLE IF NOT EXISTS corrections (
            correction_id VARCHAR PRIMARY KEY,
            workspace_id  VARCHAR NOT NULL,
            resume_id     VARCHAR NOT NULL,
            field_path    VARCHAR NOT NULL,
            before_json   VARCHAR,
            after_json    VARCHAR,
            note          VARCHAR,
            created_at    VARCHAR NOT NULL
        )
    """,
    "embeddings": """
        CREATE TABLE IF NOT EXISTS embeddings (
            embedding_id    VARCHAR PRIMARY KEY,
            workspace_id    VARCHAR NOT NULL,
            owner_type      VARCHAR NOT NULL,
            owner_id        VARCHAR NOT NULL,
            embedding_model VARCHAR NOT NULL,
            dim             INTEGER NOT NULL,
            text            VARCHAR,
            metadata_json   VARCHAR,
            created_at      VARCHAR NOT NULL
        )
    """,
}

# Table creation order respects FK-ish references (workspaces first, etc.). Not
# enforced by DuckDB DDL above (kept simple/portable) but bootstrap.py applies
# them in this order for readability and future-proofing against real FKs.
TABLE_ORDER: list[str] = [
    "workspaces", "profiles", "resumes", "candidate_records",
    "fit_runs", "corrections", "embeddings",
]
