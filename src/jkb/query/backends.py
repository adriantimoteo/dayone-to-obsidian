from __future__ import annotations

import os
from abc import ABC, abstractmethod

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"


def _raise_if_ollama_error(exc: BaseException, model: str) -> None:
    """Translate common Ollama failure modes into actionable RuntimeError messages."""
    if isinstance(exc, ConnectionRefusedError):
        raise RuntimeError("Ollama is not running. Start it with: ollama serve") from exc
    try:
        import httpx  # noqa: PLC0415
        if isinstance(exc, httpx.ConnectError):
            raise RuntimeError("Ollama is not running. Start it with: ollama serve") from exc
    except ImportError:
        pass
    try:
        import openai  # noqa: PLC0415
        if isinstance(exc, openai.NotFoundError):
            raise RuntimeError(f"Model not found. Run: ollama pull {model}") from exc
    except ImportError:
        pass


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        """Send a prompt and return the text response."""
        ...


class AnthropicBackend(LLMBackend):
    def __init__(self, client: object | None = None) -> None:
        self._client = client

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("anthropic package is not installed. Run: uv add anthropic") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAIBackend(LLMBackend):
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self._base_url = base_url
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running jkb ask --backend openai."
            )
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("openai package is not installed. Run: uv add openai") from exc
            self._client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


class OllamaBackend(OpenAIBackend):
    """LLM backend that routes requests to a local Ollama server.

    Ollama exposes an OpenAI-compatible API, so this is a thin wrapper around
    OpenAIBackend with a fixed api_key and translated error messages.

    Setup:
        Install from https://ollama.com, then:
            ollama pull llama3.2
            ollama serve          # runs automatically on Windows after install
    """

    def __init__(self, host: str = _OLLAMA_DEFAULT_HOST) -> None:
        super().__init__(base_url=f"{host}/v1", api_key="ollama")

    def complete(self, system: str, user: str, model: str, max_tokens: int) -> str:
        try:
            return super().complete(system, user, model, max_tokens)
        except Exception as exc:
            _raise_if_ollama_error(exc, model)
            raise
