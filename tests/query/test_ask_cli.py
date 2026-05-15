from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from jkb.cli import app
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
