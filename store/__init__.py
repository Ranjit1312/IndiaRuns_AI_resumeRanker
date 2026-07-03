"""store — modular, swappable embedded-DB layer. Session-only by default (HF-safe).

Three concerns are kept in separate files so the backend can later swap to an
enterprise DB without fragmenting the codebase:
  - `schema.py`     — table/row DEFINITIONS ONLY (dataclasses + DDL strings).
  - `bootstrap.py`   — DB CREATION / migration ONLY (idempotent `create_*`).
  - `*_store.py`     — CRUD/vector-search IMPLEMENTATIONS (`MemoryStore` default,
                        `DuckDBStore`/`LanceVectorStore` optional, file-backed).
  - `base.py`        — the swap seam: `RelationalStore`/`VectorStore` Protocols
                        plus the `Workspace` facade every caller actually uses.

`get_store()` is the only entry point application code needs: it returns a
`Workspace`, defaulting to a pure in-memory backend (no files, no extra deps —
safe on HF Spaces or any read-only/ephemeral filesystem). Setting
`STORE_BACKEND=duckdb+lance` (env var or `config["STORE_BACKEND"]`) switches to
file-backed persistence via DuckDB + LanceDB, which are lazy-imported only when
this path is actually selected (see `requirements-db.txt`).
"""
from __future__ import annotations

import os

from .base import RelationalStore, VectorStore, Workspace
from .memory_store import MemoryStore

__all__ = [
    "get_store", "Workspace", "RelationalStore", "VectorStore", "MemoryStore",
]

DEFAULT_DUCKDB_PATH = "data/store/app.duckdb"
DEFAULT_LANCEDB_PATH = "data/store/lancedb"


def get_store(config: dict | None = None) -> Workspace:
    """Factory: returns a `Workspace` facade, defaulting to session-only `MemoryStore`.

    `config` (optional dict) takes precedence over environment variables:
      - `STORE_BACKEND`: "memory" (default) or "duckdb+lance"
      - `DUCKDB_PATH`: path to the DuckDB file (duckdb+lance backend only)
      - `LANCEDB_PATH`: path to the LanceDB directory (duckdb+lance backend only)
    """
    cfg = config or {}
    backend = str(cfg.get("STORE_BACKEND") or os.environ.get("STORE_BACKEND") or "memory").lower()

    if backend in ("memory", "", "session"):
        mem = MemoryStore()
        return Workspace(relational=mem, vector=mem)

    if backend in ("duckdb+lance", "duckdb_lance", "duckdb"):
        # Lazy imports: only reached when a file-backed store is explicitly requested,
        # so `duckdb`/`lancedb` never need to be installed for the default path.
        from .duckdb_store import DuckDBStore
        from .lance_store import LanceVectorStore

        duckdb_path = cfg.get("DUCKDB_PATH") or os.environ.get("DUCKDB_PATH") or DEFAULT_DUCKDB_PATH
        lancedb_path = cfg.get("LANCEDB_PATH") or os.environ.get("LANCEDB_PATH") or DEFAULT_LANCEDB_PATH
        relational = DuckDBStore(duckdb_path)
        vector = LanceVectorStore(lancedb_path)
        return Workspace(relational=relational, vector=vector)

    raise ValueError(
        f"Unknown STORE_BACKEND={backend!r}; expected 'memory' or 'duckdb+lance'")
