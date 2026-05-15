from __future__ import annotations

from unittest.mock import MagicMock

from jkb.query.search import SearchResult
from jkb.query.synthesizer import (
    SynthesisResult,
    Synthesizer,
    _extract_sources,
    synthesize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_KNOW = "I don't know — the provided entries don't contain relevant information for this query."


def _make_result(
    text: str = "some text",
    entry_id: str = "2023-06-01-0900",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(text=text, metadata={}, score=score, entry_id=entry_id)


def _make_client(response_text: str) -> MagicMock:
    """Return a mock Anthropic-compatible client whose messages.create returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------


def test_extract_sources_single():
    text = "As noted in [[2023-06-01-0900]], the trip was great."
    assert _extract_sources(text) == ["[[2023-06-01-0900]]"]


def test_extract_sources_multiple():
    text = "See [[2023-06-01-0900]] and also [[2022-12-25-1200]] for details."
    assert _extract_sources(text) == ["[[2023-06-01-0900]]", "[[2022-12-25-1200]]"]


def test_extract_sources_none():
    text = "There are no wikilinks here."
    assert _extract_sources(text) == []


# ---------------------------------------------------------------------------
# SynthesisResult model
# ---------------------------------------------------------------------------


def test_synthesis_result_model():
    sr = SynthesisResult(answer="hello", sources=["[[2023-01-01-0800]]"])
    assert sr.answer == "hello"
    assert sr.sources == ["[[2023-01-01-0800]]"]


# ---------------------------------------------------------------------------
# Normal answer with citations
# ---------------------------------------------------------------------------


def test_synthesize_normal_answer_extracts_citations():
    answer_text = "You visited Tokyo in [[2019-03-15-0900]]. It was memorable."
    client = _make_client(answer_text)
    results = [_make_result(text="Visited Tokyo today.", entry_id="2019-03-15-0900")]

    synth = Synthesizer(client)
    result = synth.synthesize("When did I visit Tokyo?", results)

    assert result.answer == answer_text
    assert result.sources == ["[[2019-03-15-0900]]"]


# ---------------------------------------------------------------------------
# Empty context → I don't know (no LLM call)
# ---------------------------------------------------------------------------


def test_synthesize_empty_results_returns_no_know_without_calling_llm():
    client = _make_client("should not be called")
    synth = Synthesizer(client)
    result = synth.synthesize("anything?", [])

    assert result.answer == _NO_KNOW
    assert result.sources == []
    client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# LLM returns refusal phrase
# ---------------------------------------------------------------------------


def test_synthesize_llm_refusal_propagated():
    client = _make_client(_NO_KNOW)
    results = [_make_result(text="Unrelated text.", entry_id="2020-01-01-0800")]

    synth = Synthesizer(client)
    result = synth.synthesize("Something irrelevant?", results)

    assert result.answer == _NO_KNOW
    assert result.sources == []


# ---------------------------------------------------------------------------
# Sources list correctly populated from wikilink patterns
# ---------------------------------------------------------------------------


def test_synthesize_sources_populated_from_wikilinks():
    answer_text = (
        "The first event is in [[2021-05-10-1000]] and the second in [[2021-07-20-0800]]."
    )
    client = _make_client(answer_text)
    results = [
        _make_result(text="Event one.", entry_id="2021-05-10-1000"),
        _make_result(text="Event two.", entry_id="2021-07-20-0800"),
    ]

    result = Synthesizer(client).synthesize("Tell me about events.", results)

    assert set(result.sources) == {"[[2021-05-10-1000]]", "[[2021-07-20-0800]]"}


# ---------------------------------------------------------------------------
# Sources list empty when answer has no wikilinks
# ---------------------------------------------------------------------------


def test_synthesize_sources_empty_when_no_wikilinks():
    answer_text = "You had a great day according to the entries."
    client = _make_client(answer_text)
    results = [_make_result(entry_id="2022-03-01-0900")]

    result = Synthesizer(client).synthesize("How was my day?", results)

    assert result.sources == []


# ---------------------------------------------------------------------------
# Multiple citations in one answer
# ---------------------------------------------------------------------------


def test_synthesize_multiple_citations_all_present():
    answer_text = (
        "See [[2023-01-01-0800]], then [[2023-06-15-1200]], and again [[2023-01-01-0800]]."
    )
    client = _make_client(answer_text)
    results = [
        _make_result(entry_id="2023-01-01-0800"),
        _make_result(entry_id="2023-06-15-1200"),
    ]

    result = Synthesizer(client).synthesize("summary?", results)

    # _extract_sources returns all matches (including duplicates) — that is acceptable;
    # verify at least both are present
    assert "[[2023-01-01-0800]]" in result.sources
    assert "[[2023-06-15-1200]]" in result.sources


# ---------------------------------------------------------------------------
# Prompt construction: system message and user message content
# ---------------------------------------------------------------------------


def test_synthesize_system_message_contains_grounding_instructions():
    client = _make_client("Some answer [[2023-06-01-0900]].")
    results = [_make_result(entry_id="2023-06-01-0900")]

    Synthesizer(client).synthesize("query", results)

    call_kwargs = client.messages.create.call_args[1]
    system_msg = call_kwargs["system"]
    assert "ONLY" in system_msg or "only" in system_msg or "exclusively" in system_msg
    assert "[[entry_id]]" in system_msg or "wikilink" in system_msg.lower() or "[[" in system_msg


def test_synthesize_user_message_contains_query_and_chunk_text():
    client = _make_client("Answer [[2023-06-01-0900]].")
    chunk_text = "Had ramen for lunch."
    results = [_make_result(text=chunk_text, entry_id="2023-06-01-0900")]

    Synthesizer(client).synthesize("What did I eat?", results)

    call_kwargs = client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    assert len(messages) >= 1
    user_content = messages[0]["content"]
    assert "What did I eat?" in user_content
    assert chunk_text in user_content
    assert "2023-06-01-0900" in user_content


# ---------------------------------------------------------------------------
# Correct model and max_tokens passed to client
# ---------------------------------------------------------------------------


def test_synthesize_passes_correct_model_and_max_tokens():
    client = _make_client("answer [[e1]].")
    results = [_make_result(entry_id="e1")]

    Synthesizer(client, model="claude-haiku-4-5-20251001").synthesize("q", results)

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 1024


def test_synthesize_uses_custom_model():
    client = _make_client("answer [[e1]].")
    results = [_make_result(entry_id="e1")]

    Synthesizer(client, model="claude-opus-4-5").synthesize("q", results)

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-opus-4-5"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_convenience_synthesize_function():
    answer_text = "Result from [[2023-06-01-0900]]."
    client = _make_client(answer_text)
    results = [_make_result(entry_id="2023-06-01-0900")]

    result = synthesize("query", results, client)

    assert isinstance(result, SynthesisResult)
    assert result.answer == answer_text
    assert result.sources == ["[[2023-06-01-0900]]"]
