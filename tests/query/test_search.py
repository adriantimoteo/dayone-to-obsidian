from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from jkb.index.vectorstore import QueryResult
from jkb.query.search import HybridSearcher, SearchResult, _build_where, search
from jkb.query.parser import ParsedQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedder(vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [vector or [0.1, 0.2, 0.3, 0.4]]
    return embedder


def _make_vectorstore(results: list[QueryResult] | None = None) -> MagicMock:
    vs = MagicMock()
    vs.query.return_value = results or []
    return vs


def _query_result(
    text: str = "some text",
    entry_id: str = "entry-1",
    chunk_index: int = 0,
    distance: float = 0.2,
    metadata: dict | None = None,
) -> QueryResult:
    return QueryResult(
        text=text,
        metadata=metadata or {},
        entry_id=entry_id,
        chunk_index=chunk_index,
        distance=distance,
    )


# ---------------------------------------------------------------------------
# _build_where tests
# ---------------------------------------------------------------------------


def test_build_where_no_filters_returns_none():
    parsed = ParsedQuery()
    assert _build_where(parsed) is None


def test_build_where_date_range():
    parsed = ParsedQuery(start_date=date(2019, 1, 1), end_date=date(2019, 12, 31))
    where = _build_where(parsed)
    assert where is not None
    assert "$and" in where
    clauses = where["$and"]
    assert {"date": {"$gte": "2019-01-01"}} in clauses
    assert {"date": {"$lte": "2019-12-31"}} in clauses


def test_build_where_start_date_only():
    parsed = ParsedQuery(start_date=date(2020, 6, 1))
    where = _build_where(parsed)
    assert where == {"date": {"$gte": "2020-06-01"}}


def test_build_where_end_date_only():
    parsed = ParsedQuery(end_date=date(2020, 6, 30))
    where = _build_where(parsed)
    assert where == {"date": {"$lte": "2020-06-30"}}


def test_build_where_location():
    parsed = ParsedQuery(location="Sagada")
    where = _build_where(parsed)
    assert where == {"location_name": {"$eq": "Sagada"}}


def test_build_where_date_and_location():
    parsed = ParsedQuery(
        start_date=date(2019, 1, 1),
        end_date=date(2019, 12, 31),
        location="Tokyo",
    )
    where = _build_where(parsed)
    assert where is not None
    assert "$and" in where
    clauses = where["$and"]
    assert {"date": {"$gte": "2019-01-01"}} in clauses
    assert {"date": {"$lte": "2019-12-31"}} in clauses
    assert {"location_name": {"$eq": "Tokyo"}} in clauses


# ---------------------------------------------------------------------------
# HybridSearcher.search tests
# ---------------------------------------------------------------------------


def test_search_no_filters_passes_no_where_to_vectorstore():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    searcher.search("random thoughts", k=5)
    call_kwargs = vs.query.call_args[1] if vs.query.call_args[1] else {}
    # where should be None (not passed or passed as None)
    where_arg = call_kwargs.get("where", None)
    assert where_arg is None


def test_search_date_filter_passes_where_to_vectorstore():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    searcher.search("food in 2019", k=5)
    call_kwargs = vs.query.call_args[1] if vs.query.call_args[1] else {}
    where_arg = call_kwargs.get("where")
    assert where_arg is not None
    clauses = where_arg.get("$and", [])
    assert {"date": {"$gte": "2019-01-01"}} in clauses
    assert {"date": {"$lte": "2019-12-31"}} in clauses


def test_search_location_filter_passes_where_to_vectorstore():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    searcher.search("memories in Tokyo", k=5)
    call_kwargs = vs.query.call_args[1] if vs.query.call_args[1] else {}
    where_arg = call_kwargs.get("where")
    assert where_arg is not None
    assert where_arg == {"location_name": {"$eq": "Tokyo"}}


def test_search_returns_search_result_objects():
    embedder = _make_embedder([0.5, 0.5, 0.0, 0.0])
    raw = [_query_result(text="ate sushi", entry_id="e42", distance=0.1, metadata={"journal": "food"})]
    vs = _make_vectorstore(raw)
    searcher = HybridSearcher(vs, embedder)
    results = searcher.search("sushi", k=3)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.text == "ate sushi"
    assert r.entry_id == "e42"
    assert r.metadata == {"journal": "food"}
    assert abs(r.score - 0.9) < 1e-9


def test_search_score_is_one_minus_distance():
    embedder = _make_embedder()
    raw = [
        _query_result(distance=0.0),
        _query_result(distance=0.5, entry_id="entry-2"),
        _query_result(distance=1.0, entry_id="entry-3"),
    ]
    vs = _make_vectorstore(raw)
    searcher = HybridSearcher(vs, embedder)
    results = searcher.search("test", k=3)
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.5)
    assert results[2].score == pytest.approx(0.0)


def test_search_embeds_keywords_not_full_query_when_keywords_present():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    searcher.search("food in 2019", k=5)
    embed_call_texts = embedder.embed.call_args[0][0]
    # "food in 2019" → parser extracts 2019 as year range; keywords should be "food"
    assert embed_call_texts == ["food"]


def test_search_uses_full_query_when_no_keywords():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    # A query that is entirely consumed by date extraction (no remaining keywords)
    # "in 2019" → start/end set, keywords = ""
    searcher.search("in 2019", k=5)
    embed_call_texts = embedder.embed.call_args[0][0]
    assert embed_call_texts == ["in 2019"]


def test_search_passes_k_as_n_results():
    embedder = _make_embedder()
    vs = _make_vectorstore()
    searcher = HybridSearcher(vs, embedder)
    searcher.search("something", k=7)
    call_kwargs = vs.query.call_args[1] if vs.query.call_args[1] else {}
    assert call_kwargs.get("n_results") == 7


# ---------------------------------------------------------------------------
# Acceptance criterion: "food in 2019"
# ---------------------------------------------------------------------------


def test_acceptance_food_in_2019():
    """'food in 2019' must filter by 2019 date range AND embed the keyword 'food'."""
    embedder = _make_embedder([1.0, 0.0, 0.0, 0.0])
    food_result = _query_result(
        text="Had amazing ramen for lunch.",
        entry_id="2019-03-15",
        distance=0.05,
        metadata={"date": "2019-03-15", "journal": "personal"},
    )
    vs = _make_vectorstore([food_result])

    searcher = HybridSearcher(vs, embedder)
    results = searcher.search("food in 2019", k=10)

    # Correct where filter applied
    call_kwargs = vs.query.call_args[1] if vs.query.call_args[1] else {}
    where = call_kwargs.get("where")
    assert where is not None
    clauses = where.get("$and", [])
    assert {"date": {"$gte": "2019-01-01"}} in clauses
    assert {"date": {"$lte": "2019-12-31"}} in clauses

    # Correct keyword embedded
    embed_texts = embedder.embed.call_args[0][0]
    assert embed_texts == ["food"]

    # Correct result returned
    assert len(results) == 1
    assert results[0].text == "Had amazing ramen for lunch."
    assert results[0].entry_id == "2019-03-15"
    assert results[0].score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Convenience function `search`
# ---------------------------------------------------------------------------


def test_convenience_search_function():
    embedder = _make_embedder()
    vs = _make_vectorstore([_query_result()])
    results = search("hello", vs, embedder, k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
