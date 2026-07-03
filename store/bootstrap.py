"""DB creation / migration ONLY. Nothing else in `store/` creates tables.

Idempotent: safe to call on every process start. `duckdb`/`lancedb` are lazy
imports so this module (and the rest of the package) loads fine even when
neither library is installed — required for the `MemoryStore`-only default path
(HF Spaces / session-only mode never needs this file at all).
"""
from __future__ import annotations

import os

from .schema import TABLE_DDL, TABLE_ORDER


def create_duckdb(path: str):
    """Open (creating if needed) a DuckDB database file at `path` and ensure
    the schema exists. Returns the open connection."""
    import duckdb  # lazy import — only reached when this backend is selected

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = duckdb.connect(path)
    ensure_schema(conn)
    return conn


def create_lancedb(path: str):
    """Open (creating if needed) a LanceDB database directory at `path`.
    Table creation itself is deferred to `LanceVectorStore` (one table per
    (embedding_model, dim) pair, created lazily on first upsert) since the
    schema depends on the vector dimensionality, which isn't known here."""
    import lancedb  # lazy import — only reached when this backend is selected

    os.makedirs(path, exist_ok=True)
    return lancedb.connect(path)


def ensure_schema(conn) -> None:
    """Idempotently create every relational table defined in `schema.TABLE_DDL`,
    in dependency order. Safe to call repeatedly (`CREATE TABLE IF NOT EXISTS`)."""
    for table in TABLE_ORDER:
        conn.execute(TABLE_DDL[table])
