from __future__ import annotations
from typing import Protocol, runtime_checkable

from jkb.index.chunker import Chunk

INDEX_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str], prefix: str = INDEX_PREFIX) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        ...


class NomicEmbedder:
    """Embedder backed by nomic-ai/nomic-embed-text-v1.5 via sentence-transformers.

    Model is loaded lazily on first call to embed() so that importing this
    module does not trigger a download or slow initialisation.
    """

    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self.MODEL_NAME, trust_remote_code=True)

    def embed(self, texts: list[str], prefix: str = INDEX_PREFIX) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        prefixed = [prefix + t for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunks using the provided Embedder.
    Returns list of (chunk, embedding_vector) pairs in the same order.
    """
    if not chunks:
        return []
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts, prefix=INDEX_PREFIX)
    return list(zip(chunks, vectors))


def get_embedder(model_name: str = "nomic") -> NomicEmbedder:
    """Factory: return an Embedder for the given model name."""
    if model_name == "nomic":
        return NomicEmbedder()
    raise ValueError(
        f"Unknown embedding model: {model_name!r}. "
        "Supported models: 'nomic'. See JKB-031 for adding more."
    )
