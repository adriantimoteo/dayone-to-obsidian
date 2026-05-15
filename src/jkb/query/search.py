from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from jkb.index.embedder import QUERY_PREFIX, Embedder
from jkb.index.vectorstore import VectorStore
from jkb.query.parser import ParsedQuery, parse_query


class SearchResult(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float
    entry_id: str


def _build_where(parsed: ParsedQuery) -> dict | None:
    clauses: list[dict] = []

    if parsed.start_date is not None and parsed.end_date is not None:
        clauses.append({"date": {"$gte": parsed.start_date.isoformat()}})
        clauses.append({"date": {"$lte": parsed.end_date.isoformat()}})
    elif parsed.start_date is not None:
        clauses.append({"date": {"$gte": parsed.start_date.isoformat()}})
    elif parsed.end_date is not None:
        clauses.append({"date": {"$lte": parsed.end_date.isoformat()}})

    if parsed.location is not None:
        clauses.append({"location_name": {"$eq": parsed.location}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class HybridSearcher:
    def __init__(self, vectorstore: VectorStore, embedder: Embedder) -> None:
        self._vectorstore = vectorstore
        self._embedder = embedder

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        parsed = parse_query(query)

        text_to_embed = parsed.keywords if parsed.keywords else query
        embedding = self._embedder.embed([text_to_embed], prefix=QUERY_PREFIX)[0]

        where = _build_where(parsed)

        raw_results = self._vectorstore.query(embedding, n_results=k, where=where)

        return [
            SearchResult(
                text=r.text,
                metadata=r.metadata,
                score=1.0 - r.distance,
                entry_id=r.entry_id,
            )
            for r in raw_results
        ]


def search(query: str, vectorstore: VectorStore, embedder: Embedder, k: int = 10) -> list[SearchResult]:
    return HybridSearcher(vectorstore, embedder).search(query, k)
