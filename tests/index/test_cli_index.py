from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from jkb.cli import app
from jkb.index.pipeline import IndexStats

runner = CliRunner()


def _fake_stats() -> IndexStats:
    return IndexStats(added=3, updated=1, removed=0, skipped=5)


# ---------------------------------------------------------------------------
# 1. Valid vault path → run_index called with correct args, summary printed
# ---------------------------------------------------------------------------

def test_valid_vault_calls_run_index(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mock_run = mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mock_embedder = mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault)])

    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.args[0] == vault


# ---------------------------------------------------------------------------
# 2. Non-existent vault path → exits with code 1, error message shown
# ---------------------------------------------------------------------------

def test_nonexistent_vault_exits_1(tmp_path, mocker):
    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    missing = tmp_path / "does_not_exist"
    result = runner.invoke(app, ["index", str(missing)])

    assert result.exit_code == 1
    assert "does not exist" in result.output


# ---------------------------------------------------------------------------
# 3. --force-reindex flag → run_index called with force=True
# ---------------------------------------------------------------------------

def test_force_reindex_flag(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mock_run = mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault), "--force-reindex"])

    assert result.exit_code == 0
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs.get("force") is True


def test_no_force_reindex_flag(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mock_run = mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault), "--no-force-reindex"])

    assert result.exit_code == 0
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs.get("force") is False


# ---------------------------------------------------------------------------
# 4. --model flag → get_embedder called with the given model name
# ---------------------------------------------------------------------------

def test_model_flag_passed_to_get_embedder(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mock_get = mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault), "--model", "nomic"])

    assert result.exit_code == 0
    mock_get.assert_called_once_with("nomic")


def test_default_model_is_nomic(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mock_get = mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault)])

    assert result.exit_code == 0
    mock_get.assert_called_once_with("nomic")


# ---------------------------------------------------------------------------
# 5. --chroma-path overrides default path
# ---------------------------------------------------------------------------

def test_chroma_path_override(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()
    custom_chroma = tmp_path / "my_chroma"

    mock_store_cls = mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault), "--chroma-path", str(custom_chroma)])

    assert result.exit_code == 0
    mock_store_cls.assert_called_once_with(custom_chroma)


def test_default_chroma_path_is_vault_dot_chroma(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mock_store_cls = mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault)])

    assert result.exit_code == 0
    mock_store_cls.assert_called_once_with(vault / ".chroma")


# ---------------------------------------------------------------------------
# 6. --manifest-path overrides default path
# ---------------------------------------------------------------------------

def test_manifest_path_override(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()
    custom_manifest = tmp_path / "my-manifest.json"

    mock_manifest_cls = mocker.patch("jkb.cli.Manifest")
    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault), "--manifest-path", str(custom_manifest)])

    assert result.exit_code == 0
    mock_manifest_cls.assert_called_once_with(custom_manifest)


def test_default_manifest_path_is_vault_index_manifest(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mock_manifest_cls = mocker.patch("jkb.cli.Manifest")
    mocker.patch("jkb.cli.VectorStore")
    mocker.patch("jkb.cli.run_index", return_value=_fake_stats())
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault)])

    assert result.exit_code == 0
    mock_manifest_cls.assert_called_once_with(vault / "index-manifest.json")


# ---------------------------------------------------------------------------
# 7. Summary line format
# ---------------------------------------------------------------------------

def test_summary_line_format(tmp_path, mocker):
    vault = tmp_path / "vault"
    vault.mkdir()

    mocker.patch("jkb.cli.run_index", return_value=IndexStats(added=2, updated=4, removed=1, skipped=10))
    mocker.patch("jkb.cli.get_embedder", return_value=object())

    result = runner.invoke(app, ["index", str(vault)])

    assert result.exit_code == 0
    assert "Index complete: 2 added, 4 updated, 1 removed, 10 skipped." in result.output
