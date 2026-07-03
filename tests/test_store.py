"""Offline store/ tests (MemoryStore only — no duckdb/lancedb required).

Round-trips every entity through `get_store()`'s default (session-only, in-memory)
`Workspace` facade, and proves the vector-search isolation guarantee: two
different `embedding_model`s never bleed into each other's search results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import get_store
from store.schema import (
    CandidateRecordRow,
    CorrectionRow,
    EmbeddingRow,
    FitRunRow,
    ProfileRow,
    ResumeRow,
    Workspace as WorkspaceRow,
)

WS = "ws_test"


def _ws():
    store = get_store()   # default: MemoryStore, no config/env needed
    store.save_workspace(WorkspaceRow(workspace_id=WS, name="Test WS", created_at="t0"))
    return store


def test_default_backend_is_memory_and_workspace_round_trips():
    store = _ws()
    got = store.get_workspace(WS)
    assert got is not None
    assert got.name == "Test WS"
    assert [w.workspace_id for w in store.list_workspaces()] == [WS]


def test_profile_round_trip():
    store = _ws()
    row = ProfileRow(profile_id="p1", workspace_id=WS, name="Senior DE",
                      profile_yaml="role:\n  title: Senior DE\n", meta_yaml="",
                      source={"note": "unit test"}, created_at="t1")
    store.save_profile(row)

    fetched = store.get_profile("p1")
    assert fetched is not None
    assert fetched.name == "Senior DE"
    assert fetched.source == {"note": "unit test"}
    assert [p.profile_id for p in store.list_profiles(WS)] == ["p1"]


def test_resume_round_trip():
    store = _ws()
    row = ResumeRow(resume_id="r1", workspace_id=WS, name="XYZ Candidate",
                     raw_text="Experienced engineer...",
                     candidate_json={"candidate_id": "CAND_0000001"}, created_at="t1")
    store.save_resume(row)

    fetched = store.get_resume("r1")
    assert fetched is not None
    assert fetched.candidate_json["candidate_id"] == "CAND_0000001"
    assert [r.resume_id for r in store.list_resumes(WS)] == ["r1"]


def test_candidate_record_round_trip():
    store = _ws()
    row = CandidateRecordRow(record_id="cr1", workspace_id=WS, resume_id="r1",
                              candidate_json={"candidate_id": "CAND_0000001", "skills": []},
                              created_at="t1")
    store.save_candidate_record(row)

    fetched = store.get_candidate_record("cr1")
    assert fetched is not None
    assert fetched.resume_id == "r1"
    assert [r.record_id for r in store.list_candidate_records(WS)] == ["cr1"]


def test_fit_run_round_trip():
    store = _ws()
    row = FitRunRow(run_id="fr1", workspace_id=WS, profile_id="p1", record_id="cr1",
                     result_json={"overall": 72.5}, overall=72.5, created_at="t1")
    store.save_fit_run(row)

    runs = store.list_fit_runs(WS)
    assert len(runs) == 1
    assert runs[0].overall == 72.5
    assert runs[0].result_json == {"overall": 72.5}


def test_correction_round_trip():
    store = _ws()
    row = CorrectionRow(correction_id="c1", workspace_id=WS, resume_id="r1",
                         field_path="profile.years_of_experience", before=4, after=5,
                         note="resume said 4, candidate corrected to 5", created_at="t1")
    store.save_correction(row)

    corrections = store.list_corrections(WS)
    assert len(corrections) == 1
    assert corrections[0].before == 4
    assert corrections[0].after == 5


def test_embedding_upsert_and_nearest_neighbor_search():
    store = _ws()
    close = EmbeddingRow(embedding_id="e1", workspace_id=WS, owner_type="resume",
                          owner_id="r1", embedding_model="gemini-embedding-001", dim=3,
                          vector=[1.0, 0.0, 0.0], text="python backend engineer",
                          created_at="t1")
    far = EmbeddingRow(embedding_id="e2", workspace_id=WS, owner_type="resume",
                        owner_id="r2", embedding_model="gemini-embedding-001", dim=3,
                        vector=[0.0, 1.0, 0.0], text="marketing copywriter",
                        created_at="t1")
    store.upsert_embedding(close)
    store.upsert_embedding(far)

    results = store.search_embeddings(
        [0.9, 0.1, 0.0], embedding_model="gemini-embedding-001", dim=3, top_k=2)
    assert len(results) == 2
    top_row, top_score = results[0]
    assert top_row.embedding_id == "e1"
    assert top_score > results[1][1]


def test_embedding_models_stay_isolated():
    store = _ws()
    gemini = EmbeddingRow(embedding_id="g1", workspace_id=WS, owner_type="resume",
                           owner_id="r1", embedding_model="gemini-embedding-001", dim=3,
                           vector=[1.0, 0.0, 0.0], created_at="t1")
    gemma = EmbeddingRow(embedding_id="m1", workspace_id=WS, owner_type="resume",
                          owner_id="r1", embedding_model="embedding-gemma", dim=3,
                          vector=[1.0, 0.0, 0.0], created_at="t1")
    store.upsert_embedding(gemini)
    store.upsert_embedding(gemma)

    gemini_hits = store.search_embeddings(
        [1.0, 0.0, 0.0], embedding_model="gemini-embedding-001", dim=3, top_k=5)
    gemma_hits = store.search_embeddings(
        [1.0, 0.0, 0.0], embedding_model="embedding-gemma", dim=3, top_k=5)

    assert [r.embedding_id for r, _ in gemini_hits] == ["g1"]
    assert [r.embedding_id for r, _ in gemma_hits] == ["m1"]

    # a model/dim pair with no upserts returns empty, not a cross-model leak
    empty_hits = store.search_embeddings(
        [1.0, 0.0, 0.0], embedding_model="some-other-model", dim=3, top_k=5)
    assert empty_hits == []


def test_embedding_dim_mismatch_raises():
    store = _ws()
    bad = EmbeddingRow(embedding_id="bad1", workspace_id=WS, owner_type="resume",
                        owner_id="r1", embedding_model="gemini-embedding-001", dim=5,
                        vector=[1.0, 0.0, 0.0], created_at="t1")   # only 3 values, dim=5
    try:
        store.upsert_embedding(bad)
        assert False, "expected ValueError on vector/dim mismatch"
    except ValueError:
        pass


def test_get_store_defaults_to_memory_without_env_or_config():
    # no STORE_BACKEND env, no config -> memory backend, no duckdb/lancedb import needed
    os.environ.pop("STORE_BACKEND", None)
    store = get_store()
    assert store.vector is not None   # MemoryStore backs both relational + vector
    from store.memory_store import MemoryStore
    assert isinstance(store.relational, MemoryStore)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("all ok")
