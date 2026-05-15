from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from jkb.cli import app
from jkb.query.backends import AnthropicBackend, OllamaBackend, OpenAIBackend
from jkb.query.search import SearchResult
from jkb.query.synthesizer import SynthesisResult

runner = CliRunner()

_NO_KNOW = "I don't know — the provided entries don't contain relevant information for this query."


def _make_search_result(
    entry_id: str = "2017-04-10-0900",
    text: str = "I arrived in Sagada on a cold morning.",
    score: float = 0.91,
) -> SearchResult:
    return SearchResult(text=text, metadata={}, score=score, entry_id=entry_id)


def _make_synthesis_result(
    answer: str = "You were last in Sagada in [[2017-04-10-0900]].",
    sources: list[str] | None = None,
) -> SynthesisResult:
    return SynthesisResult(answer=answer, sources=sources or ["[[2017-04-10-0900]]"])


def _make_mock_anthropic() -> MagicMock:
    """Return a mock anthropic module with a working Anthropic() constructor."""
    mock_mod = MagicMock()
    mock_mod.Anthropic.return_value = MagicMock()
    return mock_mod


def _patch_ask(mocker, chroma_path: Path, search_results, synthesis_result):
    """Patch all heavy dependencies used by the ask command."""
    chroma_path.mkdir(parents=True, exist_ok=True)

    mocker.patch.dict("sys.modules", {"anthropic": _make_mock_anthropic()})

    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.get_embedder", return_value=MagicMock())

    mock_searcher_cls = mocker.patch("jkb.cli.HybridSearcher")
    mock_searcher_instance = MagicMock()
    mock_searcher_instance.search.return_value = search_results
    mock_searcher_cls.return_value = mock_searcher_instance

    mock_synth_cls = mocker.patch("jkb.cli.Synthesizer")
    mock_synth_instance = MagicMock()
    mock_synth_instance.synthesize.return_value = synthesis_result
    mock_synth_cls.return_value = mock_synth_instance

    return mock_searcher_cls, mock_synth_cls, mock_searcher_instance, mock_synth_instance


# ---------------------------------------------------------------------------
# 1. Basic ask: returns answer and sources
# ---------------------------------------------------------------------------


def test_basic_ask_returns_answer_and_sources(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(app, ["ask", "When was I last in Sagada?", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "You were last in Sagada" in result.output
    assert "[[2017-04-10-0900]]" in result.output


# ---------------------------------------------------------------------------
# 2. --verbose shows retrieved chunks in output
# ---------------------------------------------------------------------------


def test_verbose_flag_shows_retrieved_chunks(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result(text="I arrived in Sagada on a cold morning.")]
    synthesis = _make_synthesis_result()

    _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(
        app,
        ["ask", "When was I last in Sagada?", "--vault", str(tmp_path), "--verbose"],
    )

    assert result.exit_code == 0
    assert "2017-04-10-0900" in result.output
    assert "Sagada" in result.output


# ---------------------------------------------------------------------------
# 3. --k flag is passed through to the searcher
# ---------------------------------------------------------------------------


def test_k_flag_passed_to_searcher(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, _, mock_searcher_instance, _ = _patch_ask(mocker, chroma, results, synthesis)

    runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path), "--k", "5"])

    mock_searcher_instance.search.assert_called_once_with("test query", k=5)


# ---------------------------------------------------------------------------
# 4. --model flag is passed through to the synthesizer
# ---------------------------------------------------------------------------


def test_model_flag_passed_to_synthesizer(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    runner.invoke(
        app,
        ["ask", "test query", "--vault", str(tmp_path), "--model", "claude-opus-4-5"],
    )

    mock_synth_cls.assert_called_once()
    call_kwargs = mock_synth_cls.call_args
    assert (
        call_kwargs.kwargs.get("model") == "claude-opus-4-5"
        or (len(call_kwargs.args) > 1 and call_kwargs.args[1] == "claude-opus-4-5")
    )


# ---------------------------------------------------------------------------
# 5. Empty results from search shows "no results found" message
# ---------------------------------------------------------------------------


def test_empty_search_results_shows_no_results_message(tmp_path, mocker):
    chroma = tmp_path / ".chroma"

    _patch_ask(mocker, chroma, [], _make_synthesis_result())

    result = runner.invoke(app, ["ask", "something obscure", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "No results found" in result.output or "no results" in result.output.lower()


# ---------------------------------------------------------------------------
# 6. Sources are printed in the output
# ---------------------------------------------------------------------------


def test_sources_printed_in_output(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [
        _make_search_result(entry_id="2017-04-10-0900"),
        _make_search_result(entry_id="2018-07-22-1400", text="Back in Sagada again."),
    ]
    synthesis = _make_synthesis_result(
        answer="You visited Sagada in [[2017-04-10-0900]] and [[2018-07-22-1400]].",
        sources=["[[2017-04-10-0900]]", "[[2018-07-22-1400]]"],
    )

    _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(app, ["ask", "When was I in Sagada?", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "[[2017-04-10-0900]]" in result.output
    assert "[[2018-07-22-1400]]" in result.output


# ---------------------------------------------------------------------------
# 7. Missing chroma directory exits with helpful error
# ---------------------------------------------------------------------------


def test_missing_chroma_directory_exits_with_error(tmp_path, mocker):
    mocker.patch.dict("sys.modules", {"anthropic": _make_mock_anthropic()})

    missing_vault = tmp_path / "no_vault_here"

    result = runner.invoke(app, ["ask", "test query", "--vault", str(missing_vault)])

    assert result.exit_code == 1
    assert "does not exist" in result.output or "does not exist" in (result.stderr or "")


# ---------------------------------------------------------------------------
# 8. Default k is 10
# ---------------------------------------------------------------------------


def test_default_k_is_10(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, _, mock_searcher_instance, _ = _patch_ask(mocker, chroma, results, synthesis)

    runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path)])

    mock_searcher_instance.search.assert_called_once_with("test query", k=10)


# ---------------------------------------------------------------------------
# 9. --backend flag selects backend
# ---------------------------------------------------------------------------


def test_backend_ollama_is_default(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, OllamaBackend)


def test_backend_openai_flag(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        result = runner.invoke(
            app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "openai"]
        )

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, OpenAIBackend)


def test_backend_openai_missing_api_key_exits_1(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    chroma.mkdir(parents=True)

    with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}, clear=True):
        result = runner.invoke(
            app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "openai"]
        )

    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "OPENAI_API_KEY" in combined


def test_backend_unknown_exits_1(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    chroma.mkdir(parents=True)
    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.get_embedder", return_value=MagicMock())
    mocker.patch("jkb.cli.HybridSearcher")
    mocker.patch("jkb.cli.Synthesizer")

    result = runner.invoke(
        app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "invalid"]
    )

    assert result.exit_code == 1
    assert "unknown backend" in result.output.lower() or "unknown backend" in (result.stderr or "").lower()


# ---------------------------------------------------------------------------
# 10. --base-url is forwarded to OpenAIBackend
# ---------------------------------------------------------------------------


def test_base_url_forwarded_to_openai_backend(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        result = runner.invoke(
            app,
            [
                "ask", "test query", "--vault", str(tmp_path),
                "--backend", "openai",
                "--base-url", "https://my.endpoint/v1",
            ],
        )

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, OpenAIBackend)
    assert backend_arg._base_url == "https://my.endpoint/v1"


# ---------------------------------------------------------------------------
# 11. Ollama-specific CLI behaviour
# ---------------------------------------------------------------------------


def test_backend_anthropic_flag(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(
        app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "anthropic"]
    )

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, AnthropicBackend)


def test_backend_ollama_flag(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(
        app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "ollama"]
    )

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, OllamaBackend)


def test_ollama_host_forwarded(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    result = runner.invoke(
        app,
        ["ask", "test query", "--vault", str(tmp_path),
         "--backend", "ollama", "--ollama-host", "http://192.168.1.10:11434"],
    )

    assert result.exit_code == 0
    call_args = mock_synth_cls.call_args
    backend_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("client")
    assert isinstance(backend_arg, OllamaBackend)
    assert backend_arg._base_url == "http://192.168.1.10:11434/v1"


def test_default_model_is_llama32_for_ollama(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "ollama"])

    call_kwargs = mock_synth_cls.call_args.kwargs
    assert call_kwargs.get("model") == "llama3.2"


def test_default_model_is_haiku_for_anthropic(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]
    synthesis = _make_synthesis_result()

    _, mock_synth_cls, _, _ = _patch_ask(mocker, chroma, results, synthesis)

    runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path), "--backend", "anthropic"])

    call_kwargs = mock_synth_cls.call_args.kwargs
    assert call_kwargs.get("model") == "claude-haiku-4-5-20251001"


def test_ollama_not_running_exits_1(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]

    chroma.mkdir(parents=True)
    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.get_embedder", return_value=MagicMock())
    mock_searcher_cls = mocker.patch("jkb.cli.HybridSearcher")
    mock_searcher_cls.return_value.search.return_value = results

    mock_synth_cls = mocker.patch("jkb.cli.Synthesizer")
    mock_synth_cls.return_value.synthesize.side_effect = RuntimeError(
        "Ollama is not running. Start it with: ollama serve"
    )

    result = runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path)])

    assert result.exit_code == 1
    assert "ollama serve" in result.output or "ollama serve" in (result.stderr or "")


def test_ollama_model_not_found_exits_1(tmp_path, mocker):
    chroma = tmp_path / ".chroma"
    results = [_make_search_result()]

    chroma.mkdir(parents=True)
    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.get_embedder", return_value=MagicMock())
    mock_searcher_cls = mocker.patch("jkb.cli.HybridSearcher")
    mock_searcher_cls.return_value.search.return_value = results

    mock_synth_cls = mocker.patch("jkb.cli.Synthesizer")
    mock_synth_cls.return_value.synthesize.side_effect = RuntimeError(
        "Model not found. Run: ollama pull llama3.2"
    )

    result = runner.invoke(app, ["ask", "test query", "--vault", str(tmp_path)])

    assert result.exit_code == 1
    assert "ollama pull" in result.output or "ollama pull" in (result.stderr or "")
