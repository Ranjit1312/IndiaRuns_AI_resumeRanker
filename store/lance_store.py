"""`LanceVectorStore` — `VectorStore` implementation backed by LanceDB.

`lancedb` is imported lazily inside `__init__` so importing this module never
requires the package unless this backend is actually selected.

CRITICAL isolation rule: every embedding row carries `embedding_model` + `dim`,
and we keep **one LanceDB table per (embedding_model, dim) pair**. Different
embedding models (e.g. `gemini-embedding-001` at 3072-dim vs a local
EmbeddingGemma at 768-dim) are never stored in — or searched across — the same
table, so cosine similarity is never computed between incompatible vector spaces.
"""
from __future__ import annotations

import re

from .schema import EmbeddingRow


def _table_name(embedding_model: str, dim: int) -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_]", "_", embedding_model)
    return f"emb_{safe_model}_{dim}"


class LanceVectorStore:
    """Implements the `VectorStore` Protocol against a LanceDB directory."""

    def __init__(self, path: str):
        import lancedb  # lazy import — only reached when this backend is selected

        self._path = path
        self._db = lancedb.connect(path)
        self._tables: dict[str, object] = {}

    def _get_or_create_table(self, embedding_model: str, dim: int):
        name = _table_name(embedding_model, dim)
        if name in self._tables:
            return self._tables[name]
        if name in self._db.table_names():
            tbl = self._db.open_table(name)
        else:
            sample = [{
                "embedding_id": "__init__", "workspace_id": "", "owner_type": "",
                "owner_id": "", "embedding_model": embedding_model, "dim": dim,
                "text": "", "metadata_json": "{}", "created_at": "",
                "vector": [0.0] * dim,
            }]
            tbl = self._db.create_table(name, data=sample)
            tbl.delete("embedding_id = '__init__'")
        self._tables[name] = tbl
        return tbl

    def upsert_embedding(self, row: EmbeddingRow) -> None:
        import json

        if len(row.vector) != row.dim:
            raise ValueError(
                f"embedding vector length {len(row.vector)} != declared dim {row.dim}")
        tbl = self._get_or_create_table(row.embedding_model, row.dim)
        tbl.delete(f"embedding_id = '{row.embedding_id}'")
        tbl.add([{
            "embedding_id": row.embedding_id, "workspace_id": row.workspace_id,
            "owner_type": row.owner_type, "owner_id": row.owner_id,
            "embedding_model": row.embedding_model, "dim": row.dim,
            "text": row.text, "metadata_json": json.dumps(row.metadata),
            "created_at": row.created_at, "vector": list(row.vector),
        }])

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
        import json

        name = _table_name(embedding_model, dim)
        if name not in self._db.table_names():
            return []
        tbl = self._get_or_create_table(embedding_model, dim)
        q = tbl.search(list(query_vector)).limit(max(top_k * 4, top_k))
        results = q.to_list()

        out: list[tuple[EmbeddingRow, float]] = []
        for r in results:
            if workspace_id is not None and r.get("workspace_id") != workspace_id:
                continue
            if owner_type is not None and r.get("owner_type") != owner_type:
                continue
            dist = r.get("_distance", 0.0)
            # LanceDB's default metric is L2 on the raw vectors; convert to a
            # cosine-similarity-like score assuming normalized query/index vectors
            # (cosine = 1 - L2^2/2 for unit vectors). Callers that pre-normalize
            # embeddings (as harness.backends does) get a true cosine score.
            similarity = 1.0 - (dist / 2.0)
            row = EmbeddingRow(
                embedding_id=r["embedding_id"], workspace_id=r["workspace_id"],
                owner_type=r["owner_type"], owner_id=r["owner_id"],
                embedding_model=r["embedding_model"], dim=r["dim"],
                vector=list(r.get("vector", [])), text=r.get("text", ""),
                metadata=json.loads(r.get("metadata_json") or "{}"),
                created_at=r.get("created_at", ""))
            out.append((row, similarity))

        out.sort(key=lambda pair: pair[1], reverse=True)
        return out[:top_k]
