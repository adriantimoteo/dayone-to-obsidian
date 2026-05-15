from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from jkb.index.chunker import Chunk

COLLECTION_NAME = "jkb_entries"

_METADATA_FIELDS = ("date", "location_name", "coords", "journal", "tags", "starred", "activity", "device")


@dataclass
class QueryResult:
    text: str
    metadata: dict[str, Any]
    entry_id: str
    chunk_index: int
    distance: float


def _build_chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata
    out: dict[str, Any] = {"entry_id": chunk.entry_id, "chunk_index": chunk.chunk_index}
    for field in _METADATA_FIELDS:
        val = meta.get(field)
        if val is None:
            continue
        if field == "tags":
            # ChromaDB only stores strings/numbers; serialize list as pipe-separated
            out[field] = "|".join(val) if isinstance(val, list) else str(val)
        elif field == "coords":
            # [lat, lon] list → "lat,lon" string
            out[field] = ",".join(str(x) for x in val) if isinstance(val, list) else str(val)
        elif field == "starred":
            out[field] = int(bool(val))
        else:
            out[field] = val
    return out


def _parse_chroma_metadata(raw: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
    entry_id = raw.get("entry_id", "")
    chunk_index = int(raw.get("chunk_index", 0))
    meta: dict[str, Any] = {}
    for field in _METADATA_FIELDS:
        if field not in raw:
            continue
        val = raw[field]
        if field == "tags":
            meta[field] = val.split("|") if val else []
        elif field == "coords":
            meta[field] = [float(x) for x in val.split(",")] if val else []
        elif field == "starred":
            meta[field] = bool(val)
        else:
            meta[field] = val
    return meta, entry_id, chunk_index


class VectorStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        _client: chromadb.ClientAPI | None = None,
        _collection_name: str = COLLECTION_NAME,
    ) -> None:
        if _client is not None:
            self._client = _client
        else:
            if path is None:
                raise ValueError("path is required when _client is not provided")
            self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def ephemeral(cls, *, _collection_name: str = COLLECTION_NAME) -> VectorStore:
        return cls(_client=chromadb.EphemeralClient(), _collection_name=_collection_name)

    def add(self, chunks_and_embeddings: list[tuple[Chunk, list[float]]]) -> None:
        if not chunks_and_embeddings:
            return
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for chunk, embedding in chunks_and_embeddings:
            ids.append(f"{chunk.entry_id}_{chunk.chunk_index}")
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append(_build_chroma_metadata(chunk))
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, entry_id: str) -> None:
        self._collection.delete(where={"entry_id": entry_id})

    def query(
        self,
        embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[QueryResult]:
        count = self._collection.count()
        if count == 0:
            return []
        # cap n_results so ChromaDB doesn't error when asking for more than available
        actual_n = min(n_results, count)
        kwargs: dict[str, Any] = {"query_embeddings": [embedding], "n_results": actual_n}
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out: list[QueryResult] = []
        ids_row = result["ids"][0]
        docs_row = result["documents"][0]
        metas_row = result["metadatas"][0]
        dists_row = result["distances"][0]
        for doc, raw_meta, dist in zip(docs_row, metas_row, dists_row):
            meta, entry_id, chunk_index = _parse_chroma_metadata(raw_meta)
            out.append(QueryResult(
                text=doc,
                metadata=meta,
                entry_id=entry_id,
                chunk_index=chunk_index,
                distance=dist,
            ))
        return out
