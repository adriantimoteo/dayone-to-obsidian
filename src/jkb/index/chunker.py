from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

TOKEN_LIMIT = 300  # chunks stay below this token count


@dataclass
class Chunk:
    text: str            # chunk body content (no model prefix)
    metadata: dict[str, Any]  # full frontmatter metadata dict
    chunk_index: int     # 0-based index within the entry
    entry_id: str        # entry UUID (from metadata.get("uuid", ""))


def chunk_entry(body_text: str, metadata: dict[str, Any]) -> list[Chunk]:
    """
    Split a journal entry body into Chunks.

    - If the entire body is under TOKEN_LIMIT tokens: single chunk.
    - Otherwise: split on paragraph boundaries (double newline), accumulate
      paragraphs greedily until adding the next would exceed TOKEN_LIMIT,
      then start a new chunk. A single paragraph that exceeds TOKEN_LIMIT on
      its own becomes its own chunk rather than being split mid-sentence.

    metadata must include "uuid" for entry_id; other keys are passed through.
    """
    entry_id = str(metadata.get("uuid", ""))
    text = body_text.strip()

    if not text:
        return [Chunk(text="", metadata=metadata, chunk_index=0, entry_id=entry_id)]

    if _count_tokens(text) <= TOKEN_LIMIT:
        return [Chunk(text=text, metadata=metadata, chunk_index=0, entry_id=entry_id)]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if current_parts and current_tokens + para_tokens > TOKEN_LIMIT:
            # flush current accumulation
            chunks.append(Chunk(
                text="\n\n".join(current_parts),
                metadata=metadata,
                chunk_index=len(chunks),
                entry_id=entry_id,
            ))
            current_parts = []
            current_tokens = 0
        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append(Chunk(
            text="\n\n".join(current_parts),
            metadata=metadata,
            chunk_index=len(chunks),
            entry_id=entry_id,
        ))

    return chunks


def _count_tokens(text: str) -> int:
    """Word-based token approximation (avoids model dependency at chunk time)."""
    return int(len(text.split()) * 1.3)
