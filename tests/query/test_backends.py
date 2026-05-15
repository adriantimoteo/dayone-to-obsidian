from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jkb.query.backends import AnthropicBackend, LLMBackend, OllamaBackend, OpenAIBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_client(response_text: str) -> MagicMock:
    content_block = MagicMock()
    content_block.text = response_text
    message = MagicMock()
    message.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = message
    return client


def _make_openai_client(response_text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = response_text
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# LLMBackend ABC
# ---------------------------------------------------------------------------

def test_llm_backend_is_abstract():
    with pytest.raises(TypeError):
        LLMBackend()  # type: ignore[abstract]


def test_anthropic_backend_is_llm_backend():
    assert isinstance(AnthropicBackend(), LLMBackend)


def test_openai_backend_is_llm_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(OpenAIBackend(), LLMBackend)


# ---------------------------------------------------------------------------
# AnthropicBackend
# ---------------------------------------------------------------------------

def test_anthropic_backend_complete_calls_messages_create():
    client = _make_anthropic_client("the answer")
    backend = AnthropicBackend(client=client)
    result = backend.complete(system="sys", user="usr", model="claude-haiku-4-5-20251001", max_tokens=512)
    assert result == "the answer"
    client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="sys",
        messages=[{"role": "user", "content": "usr"}],
    )


def test_anthropic_backend_passes_model_and_max_tokens():
    client = _make_anthropic_client("ok")
    AnthropicBackend(client=client).complete(system="s", user="u", model="claude-opus-4-7", max_tokens=2048)
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-opus-4-7"
    assert call_kwargs["max_tokens"] == 2048


def test_anthropic_backend_raises_import_error_when_package_missing():
    backend = AnthropicBackend()
    with patch.dict("sys.modules", {"anthropic": None}):
        with pytest.raises(ImportError, match="uv add anthropic"):
            backend.complete(system="s", user="u", model="m", max_tokens=100)


def test_anthropic_backend_lazy_loads_client():
    backend = AnthropicBackend()
    assert backend._client is None


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------

def test_openai_backend_complete_puts_system_in_messages(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = _make_openai_client("openai answer")

    with patch("openai.OpenAI", return_value=client):
        backend = OpenAIBackend()
        result = backend.complete(system="sys prompt", user="user prompt", model="gpt-4o-mini", max_tokens=256)

    assert result == "openai answer"
    call_kwargs = client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys prompt"}
    assert messages[1] == {"role": "user", "content": "user prompt"}


def test_openai_backend_passes_model_and_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = _make_openai_client("ok")

    with patch("openai.OpenAI", return_value=client):
        backend = OpenAIBackend()
        backend.complete(system="s", user="u", model="gpt-4o", max_tokens=1024)

    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["max_tokens"] == 1024


def test_openai_backend_raises_value_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIBackend()


def test_openai_backend_raises_import_error_when_package_missing():
    backend = OpenAIBackend(api_key="sk-test")
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="uv add openai"):
            backend.complete(system="s", user="u", model="m", max_tokens=100)


def test_openai_backend_passes_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = _make_openai_client("ok")

    with patch("openai.OpenAI", return_value=client) as mock_cls:
        backend = OpenAIBackend(base_url="https://custom.endpoint/v1")
        backend.complete(system="s", user="u", model="m", max_tokens=100)

    mock_cls.assert_called_once_with(base_url="https://custom.endpoint/v1", api_key="sk-test")


def test_openai_backend_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _make_openai_client("ok")

    with patch("openai.OpenAI", return_value=client) as mock_cls:
        backend = OpenAIBackend(api_key="explicit-key")
        backend.complete(system="s", user="u", model="m", max_tokens=100)

    mock_cls.assert_called_once_with(base_url=None, api_key="explicit-key")


def test_openai_backend_lazy_loads_openai_client():
    backend = OpenAIBackend(api_key="sk-test")
    assert backend._client is None


# ---------------------------------------------------------------------------
# OllamaBackend
# ---------------------------------------------------------------------------

def test_ollama_backend_is_llm_backend():
    assert isinstance(OllamaBackend(), LLMBackend)


def test_ollama_backend_is_openai_backend_subclass():
    assert isinstance(OllamaBackend(), OpenAIBackend)


def test_ollama_backend_default_host():
    backend = OllamaBackend()
    assert backend._base_url == "http://localhost:11434/v1"
    assert backend._api_key == "ollama"


def test_ollama_backend_custom_host():
    backend = OllamaBackend(host="http://192.168.1.10:11434")
    assert backend._base_url == "http://192.168.1.10:11434/v1"


def test_ollama_backend_complete_calls_chat_completions():
    client = _make_openai_client("local answer")

    with patch("openai.OpenAI", return_value=client):
        backend = OllamaBackend()
        result = backend.complete(system="sys", user="usr", model="llama3.2", max_tokens=512)

    assert result == "local answer"
    client.chat.completions.create.assert_called_once()
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "llama3.2"


def test_ollama_backend_translates_connect_error():
    backend = OllamaBackend()

    class _FakeConnectError(Exception):
        pass

    with patch("httpx.ConnectError", _FakeConnectError):
        with patch.object(
            backend, "_get_client",
            side_effect=_FakeConnectError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="ollama serve"):
                backend.complete(system="s", user="u", model="llama3.2", max_tokens=100)


def test_ollama_backend_translates_connection_refused_error():
    backend = OllamaBackend()
    with patch.object(
        backend, "_get_client",
        side_effect=ConnectionRefusedError("refused"),
    ):
        with pytest.raises(RuntimeError, match="ollama serve"):
            backend.complete(system="s", user="u", model="llama3.2", max_tokens=100)


def test_ollama_backend_translates_model_not_found():
    backend = OllamaBackend()

    class _FakeNotFoundError(Exception):
        pass

    with patch("openai.NotFoundError", _FakeNotFoundError):
        with patch.object(
            backend, "_get_client",
            side_effect=_FakeNotFoundError("404"),
        ):
            with pytest.raises(RuntimeError, match="ollama pull llama3.2"):
                backend.complete(system="s", user="u", model="llama3.2", max_tokens=100)
