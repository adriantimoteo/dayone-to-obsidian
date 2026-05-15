from __future__ import annotations
import pytest
from jkb.index.chunker import Chunk, TOKEN_LIMIT, chunk_entry, _count_tokens


def _make_paragraph(n_words: int) -> str:
    """Build a paragraph of n_words distinct words."""
    return " ".join(f"word{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# 1. Short entry → single chunk
# ---------------------------------------------------------------------------
def test_short_entry_single_chunk():
    meta = {"uuid": "abc-123", "date": "2024-01-01"}
    body = "This is a short journal entry that is well under the token limit."
    chunks = chunk_entry(body, meta)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata == meta
    assert chunks[0].entry_id == "abc-123"
    assert chunks[0].text == body.strip()


# ---------------------------------------------------------------------------
# 2. Empty body → single chunk with empty text
# ---------------------------------------------------------------------------
def test_empty_body_single_chunk():
    meta = {"uuid": "empty-uuid"}
    chunks = chunk_entry("", meta)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].chunk_index == 0
    assert chunks[0].entry_id == "empty-uuid"


# ---------------------------------------------------------------------------
# 3. Long entry splits at paragraph boundaries
# ---------------------------------------------------------------------------
def test_long_entry_splits_at_paragraphs():
    # 4 paragraphs of 80 words each → ~104 tokens each, total ~416 tokens
    paras = [_make_paragraph(80) for _ in range(4)]
    body = "\n\n".join(paras)
    meta = {"uuid": "long-uuid"}

    chunks = chunk_entry(body, meta)

    assert len(chunks) > 1, "Long entry must produce more than 1 chunk"

    # No chunk should exceed TOKEN_LIMIT tokens
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= TOKEN_LIMIT, (
            f"Chunk {chunk.chunk_index} exceeds TOKEN_LIMIT: "
            f"{_count_tokens(chunk.text)} tokens"
        )

    # All paragraphs must be present across all chunks
    combined = "\n\n".join(c.text for c in chunks)
    for para in paras:
        assert para in combined, "Paragraph missing from chunked output"

    # chunk_index must be sequential from 0
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# 4. Single oversized paragraph → 1 chunk (can't split without mid-sentence cut)
# ---------------------------------------------------------------------------
def test_single_oversized_paragraph():
    # ~520 tokens (400 words * 1.3), no double newlines
    body = _make_paragraph(400)
    meta = {"uuid": "oversized-uuid"}
    chunks = chunk_entry(body, meta)
    assert len(chunks) == 1
    assert chunks[0].text == body


# ---------------------------------------------------------------------------
# 5. Metadata preserved on all chunks
# ---------------------------------------------------------------------------
def test_chunk_metadata_preserved():
    meta = {"uuid": "meta-uuid", "mood": "happy", "tags": ["a", "b"], "rating": 5}
    # Build a body long enough to produce multiple chunks
    paras = [_make_paragraph(80) for _ in range(4)]
    body = "\n\n".join(paras)

    chunks = chunk_entry(body, meta)
    assert len(chunks) > 1, "Need multiple chunks for this test to be meaningful"
    for chunk in chunks:
        assert chunk.metadata == meta


# ---------------------------------------------------------------------------
# 6. chunk_index is sequential
# ---------------------------------------------------------------------------
def test_chunk_index_sequential():
    paras = [_make_paragraph(80) for _ in range(6)]
    body = "\n\n".join(paras)
    meta = {"uuid": "seq-uuid"}

    chunks = chunk_entry(body, meta)
    assert len(chunks) >= 3, "Need at least 3 chunks for this test"
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# 7. entry_id comes from uuid field
# ---------------------------------------------------------------------------
def test_entry_id_from_uuid():
    meta = {"uuid": "MYUUID", "extra": "data"}
    paras = [_make_paragraph(80) for _ in range(4)]
    body = "\n\n".join(paras)

    chunks = chunk_entry(body, meta)
    for chunk in chunks:
        assert chunk.entry_id == "MYUUID"


# ---------------------------------------------------------------------------
# 8. Missing uuid → entry_id is "" (no crash)
# ---------------------------------------------------------------------------
def test_entry_id_missing_uuid():
    meta = {"date": "2024-01-01"}  # no "uuid" key
    body = "Short body text."
    chunks = chunk_entry(body, meta)
    assert len(chunks) == 1
    assert chunks[0].entry_id == ""
