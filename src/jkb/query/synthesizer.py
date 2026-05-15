from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from jkb.query.search import SearchResult

_SYSTEM_PROMPT = """\
You are a personal journal assistant. Answer the user's question ONLY using the provided journal entry context below.

Rules:
1. Base your answer exclusively on the provided journal entries. Do not use outside knowledge.
2. Cite every journal entry you draw from using its Obsidian wikilink: [[entry_id]].
3. If the provided entries do not contain information relevant to the question, respond with exactly:
   "I don't know — the provided entries don't contain relevant information for this query."
4. Do not speculate or add information not present in the entries.\
"""

_NO_CONTEXT_ANSWER = (
    "I don't know — the provided entries don't contain relevant information for this query."
)


def _format_context(results: list[SearchResult]) -> str:
    parts: list[str] = []
    for r in results:
        parts.append(f"Entry [[{r.entry_id}]]:\n{r.text}\n---")
    return "\n".join(parts)


def _extract_sources(text: str) -> list[str]:
    return re.findall(r"\[\[[^\[\]]+\]\]", text)


class SynthesisResult(BaseModel):
    answer: str
    sources: list[str]


class Synthesizer:
    def __init__(self, client: Any, model: str = "claude-haiku-4-5-20251001") -> None:
        self._client = client
        self._model = model

    def synthesize(self, query: str, results: list[SearchResult]) -> SynthesisResult:
        if not results:
            return SynthesisResult(answer=_NO_CONTEXT_ANSWER, sources=[])

        context = _format_context(results)
        user_message = f"Question: {query}\n\nJournal entries:\n{context}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        sources = _extract_sources(answer)
        return SynthesisResult(answer=answer, sources=sources)


def synthesize(
    query: str,
    results: list[SearchResult],
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
) -> SynthesisResult:
    return Synthesizer(client, model).synthesize(query, results)
