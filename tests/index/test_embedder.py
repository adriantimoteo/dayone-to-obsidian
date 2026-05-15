from __future__ import annotations
import pytest
from jkb.index.chunker import Chunk
from jkb.index.embedder import (
    INDEX_PREFIX,
    QUERY_PREFIX,
    NomicEmbedder,
    embed_chunks,
    get_embedder,
)


# ---------------------------------------------------------------------------
# Shared fake embedder
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    def embed(self, texts: list[str], prefix: str = "") -> list[list[float]]:
        return [[float(i)] * 4 for i in range(len(texts))]


def _make_chunk(text: str = "hello world", uuid: str = "test-uuid", idx: int = 0) -> Chunk:
    return Chunk(text=text, metadata={"uuid": uuid}, chunk_index=idx, entry_id=uuid)


# ---------------------------------------------------------------------------
# 1. embed_chunks returns (chunk, vector) pairs
# ---------------------------------------------------------------------------
def test_embed_chunks_returns_pairs():
    chunk = _make_chunk()

    class _SingleEmbedder:
        def embed(self, texts, prefix=""):
            return [[0.1] * 5]

    result = embed_chunks([chunk], _SingleEmbedder())
    assert result == [(chunk, [0.1] * 5)]


# ---------------------------------------------------------------------------
# 2. embed_chunks with empty list returns []
# ---------------------------------------------------------------------------
def test_embed_chunks_empty_list():
    result = embed_chunks([], _FakeEmbedder())
    assert result == []


# ---------------------------------------------------------------------------
# 3. embed_chunks preserves order
# ---------------------------------------------------------------------------
def test_embed_chunks_order_preserved():
    chunks = [_make_chunk(text=f"text {i}", idx=i) for i in range(3)]
    result = embed_chunks(chunks, _FakeEmbedder())

    assert len(result) == 3
    for i, (chunk, vec) in enumerate(result):
        assert chunk is chunks[i], f"Chunk at position {i} is out of order"
        # _FakeEmbedder returns [[0.0]*4, [1.0]*4, [2.0]*4]
        assert vec == [float(i)] * 4


# ---------------------------------------------------------------------------
# 4. get_embedder("nomic") returns NomicEmbedder
# ---------------------------------------------------------------------------
def test_get_embedder_nomic_returns_nomic():
    embedder = get_embedder("nomic")
    assert isinstance(embedder, NomicEmbedder)


# ---------------------------------------------------------------------------
# 5. get_embedder with unknown name raises ValueError
# ---------------------------------------------------------------------------
def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_embedder("gpt4")


# ---------------------------------------------------------------------------
# 6. NomicEmbedder._model is None before first embed call (lazy loading)
# ---------------------------------------------------------------------------
def test_nomic_embedder_lazy_load():
    embedder = NomicEmbedder()
    assert embedder._model is None, "_model should not be loaded until embed() is called"


# ---------------------------------------------------------------------------
# 7. Prefix constants have correct values
# ---------------------------------------------------------------------------
def test_index_prefix_constant():
    assert INDEX_PREFIX == "search_document: "
    assert QUERY_PREFIX == "search_query: "
