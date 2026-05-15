from __future__ import annotations

import uuid

from jkb.index.chunker import Chunk
from jkb.index.vectorstore import VectorStore, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, entry_id: str = "entry-1", chunk_index: int = 0, **meta_kwargs) -> Chunk:
    metadata: dict = {"uuid": entry_id, **meta_kwargs}
    return Chunk(text=text, metadata=metadata, chunk_index=chunk_index, entry_id=entry_id)


def _store() -> VectorStore:
    # Each call gets a unique collection so tests don't bleed into each other
    # (EphemeralClient is a process-level singleton in chromadb 1.x)
    return VectorStore.ephemeral(_collection_name=f"test_{uuid.uuid4().hex}")


# Short 4-dim vectors for tests
ZERO_VEC = [0.0, 0.0, 0.0, 0.0]
A_VEC = [1.0, 0.0, 0.0, 0.0]
B_VEC = [0.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Test 1: add then query returns the inserted chunk text
# ---------------------------------------------------------------------------

def test_add_then_query_returns_text():
    store = _store()
    chunk = _chunk("Hello world", entry_id="e1")
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    assert len(results) == 1
    assert results[0].text == "Hello world"
    assert results[0].entry_id == "e1"
    assert results[0].chunk_index == 0
    assert isinstance(results[0].distance, float)


# ---------------------------------------------------------------------------
# Test 2: add is idempotent — same chunks twice doesn't duplicate
# ---------------------------------------------------------------------------

def test_add_is_idempotent():
    store = _store()
    chunk = _chunk("Idempotent entry", entry_id="e2")
    store.add([(chunk, A_VEC)])
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=10)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Test 3: delete removes all chunks for an entry_id
# ---------------------------------------------------------------------------

def test_delete_removes_entry_chunks():
    store = _store()
    c0 = _chunk("chunk 0", entry_id="e3", chunk_index=0)
    c1 = _chunk("chunk 1", entry_id="e3", chunk_index=1)
    other = _chunk("other entry", entry_id="e99", chunk_index=0)
    store.add([(c0, A_VEC), (c1, A_VEC), (other, B_VEC)])
    store.delete("e3")
    results = store.query(A_VEC, n_results=10)
    assert all(r.entry_id != "e3" for r in results)
    assert any(r.entry_id == "e99" for r in results)


# ---------------------------------------------------------------------------
# Test 4: query with where filter returns only matching metadata
# ---------------------------------------------------------------------------

def test_query_with_where_filter():
    store = _store()
    journal_a = _chunk("Morning thoughts", entry_id="e4", chunk_index=0, journal="personal")
    journal_b = _chunk("Work standup", entry_id="e5", chunk_index=0, journal="work")
    store.add([(journal_a, A_VEC), (journal_b, A_VEC)])
    results = store.query(A_VEC, n_results=5, where={"journal": "personal"})
    assert len(results) == 1
    assert results[0].metadata["journal"] == "personal"
    assert results[0].entry_id == "e4"


# ---------------------------------------------------------------------------
# Test 5: query on empty store returns []
# ---------------------------------------------------------------------------

def test_query_empty_store_returns_empty():
    store = _store()
    results = store.query(A_VEC, n_results=5)
    assert results == []


# ---------------------------------------------------------------------------
# Test 6: metadata round-trip — tags list serialized/deserialized correctly
# ---------------------------------------------------------------------------

def test_tags_round_trip():
    store = _store()
    chunk = _chunk("Tagged entry", entry_id="e6", tags=["travel", "hiking", "mountains"])
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    assert len(results) == 1
    assert results[0].metadata["tags"] == ["travel", "hiking", "mountains"]


# ---------------------------------------------------------------------------
# Test 6b: empty tags list round-trips correctly
# ---------------------------------------------------------------------------

def test_empty_tags_round_trip():
    store = _store()
    chunk = _chunk("No tags", entry_id="e6b", tags=[])
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    assert results[0].metadata["tags"] == []


# ---------------------------------------------------------------------------
# Test 7: multi-entry store — query returns nearest result
# ---------------------------------------------------------------------------

def test_query_returns_nearest_result():
    store = _store()
    # entry "near" has vector A_VEC (same as query)
    # entry "far" has vector B_VEC (orthogonal to query)
    near = _chunk("Near entry", entry_id="near")
    far = _chunk("Far entry", entry_id="far")
    store.add([(near, A_VEC), (far, B_VEC)])
    results = store.query(A_VEC, n_results=2)
    assert len(results) == 2
    # nearest result should be the "near" entry (distance ~0)
    assert results[0].entry_id == "near"
    assert results[0].distance < results[1].distance


# ---------------------------------------------------------------------------
# Test 8: QueryResult is a proper dataclass with expected fields
# ---------------------------------------------------------------------------

def test_query_result_fields():
    store = _store()
    chunk = _chunk("Field check", entry_id="e8", journal="test", starred=True)
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    r = results[0]
    assert r.text == "Field check"
    assert r.entry_id == "e8"
    assert r.chunk_index == 0
    assert r.metadata["journal"] == "test"
    assert r.metadata["starred"] is True
    assert isinstance(r.distance, float)


# ---------------------------------------------------------------------------
# Test 9: None metadata values are not stored (no crash on round-trip)
# ---------------------------------------------------------------------------

def test_none_metadata_values_omitted():
    store = _store()
    chunk = Chunk(
        text="Sparse metadata",
        metadata={"uuid": "e9", "journal": "personal", "tags": None, "location_name": None},
        chunk_index=0,
        entry_id="e9",
    )
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    assert results[0].text == "Sparse metadata"
    assert "tags" not in results[0].metadata
    assert "location_name" not in results[0].metadata


# ---------------------------------------------------------------------------
# Test 10: coords round-trip — [lat, lon] list serialized/deserialized correctly
# ---------------------------------------------------------------------------

def test_coords_round_trip():
    store = _store()
    chunk = _chunk("Location entry", entry_id="e10", coords=[14.5547, 121.0244])
    store.add([(chunk, A_VEC)])
    results = store.query(A_VEC, n_results=1)
    assert results[0].metadata["coords"] == [14.5547, 121.0244]
