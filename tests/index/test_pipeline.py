from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from jkb.index.manifest import Manifest
from jkb.index.pipeline import IndexStats, run_index
from jkb.index.vectorstore import VectorStore


class _FakeEmbedder:
    def embed(self, texts, prefix=""):
        return [[0.1] * 4 for _ in texts]


def _store() -> VectorStore:
    return VectorStore.ephemeral(_collection_name=f"test_{uuid.uuid4().hex}")


def _write_md(path: Path, body: str, frontmatter: str | None = None) -> None:
    if frontmatter is not None:
        path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    else:
        path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. New vault file is indexed (added=1, skipped=0)
# ---------------------------------------------------------------------------

def test_new_file_is_added(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_md(vault / "entry.md", "Hello world", frontmatter="uuid: abc-001\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest, _FakeEmbedder())

    assert stats.added == 1
    assert stats.skipped == 0
    assert stats.updated == 0
    assert stats.removed == 0


# ---------------------------------------------------------------------------
# 2. Unchanged file on second run is skipped (added=0, skipped=1)
# ---------------------------------------------------------------------------

def test_unchanged_file_is_skipped(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_md(vault / "entry.md", "Hello world", frontmatter="uuid: abc-002\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")

    run_index(vault, store, manifest, _FakeEmbedder())

    manifest2 = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest2, _FakeEmbedder())

    assert stats.added == 0
    assert stats.skipped == 1
    assert stats.updated == 0
    assert stats.removed == 0


# ---------------------------------------------------------------------------
# 3. Modified file on second run is re-indexed (updated=1)
# ---------------------------------------------------------------------------

def test_modified_file_is_updated(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    entry = vault / "entry.md"
    _write_md(entry, "Original content", frontmatter="uuid: abc-003\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    run_index(vault, store, manifest, _FakeEmbedder())

    _write_md(entry, "Updated content", frontmatter="uuid: abc-003\ndate: 2024-01-01")

    manifest2 = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest2, _FakeEmbedder())

    assert stats.updated == 1
    assert stats.added == 0
    assert stats.skipped == 0


# ---------------------------------------------------------------------------
# 4. Deleted file is removed from store and manifest (removed=1)
# ---------------------------------------------------------------------------

def test_deleted_file_is_removed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    entry = vault / "entry.md"
    _write_md(entry, "Some content", frontmatter="uuid: abc-004\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    run_index(vault, store, manifest, _FakeEmbedder())

    entry.unlink()

    manifest2 = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest2, _FakeEmbedder())

    assert stats.removed == 1
    assert stats.added == 0
    assert stats.skipped == 0

    # The path should no longer be in the manifest
    manifest2_check = Manifest(tmp_path / "index-manifest.json")
    manifest2_check.load()
    assert "entry.md" not in manifest2_check.all_paths()


# ---------------------------------------------------------------------------
# 5. force=True re-indexes unchanged files
# ---------------------------------------------------------------------------

def test_force_reindexes_unchanged(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_md(vault / "entry.md", "Static content", frontmatter="uuid: abc-005\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    run_index(vault, store, manifest, _FakeEmbedder())

    manifest2 = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest2, _FakeEmbedder(), force=True)

    assert stats.updated == 1
    assert stats.skipped == 0
    assert stats.added == 0


# ---------------------------------------------------------------------------
# 6. File without frontmatter doesn't crash (uuid falls back to rel_path)
# ---------------------------------------------------------------------------

def test_file_without_frontmatter(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "no-frontmatter.md").write_text("Just plain text here.", encoding="utf-8")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest, _FakeEmbedder())

    assert stats.added == 1
    assert stats.skipped == 0

    manifest.load()
    assert "no-frontmatter.md" in manifest.all_paths()


# ---------------------------------------------------------------------------
# 7. IndexStats counts are correct across a mixed run (new + unchanged + deleted)
# ---------------------------------------------------------------------------

def test_mixed_run_stats(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create two files for the first run
    _write_md(vault / "keep.md", "Keep this", frontmatter="uuid: abc-007a\ndate: 2024-01-01")
    to_delete = vault / "delete.md"
    _write_md(to_delete, "Delete this", frontmatter="uuid: abc-007b\ndate: 2024-01-01")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    first = run_index(vault, store, manifest, _FakeEmbedder())
    assert first.added == 2

    # Second run: keep.md unchanged, delete.md removed, new.md added
    to_delete.unlink()
    _write_md(vault / "new.md", "Brand new", frontmatter="uuid: abc-007c\ndate: 2024-01-01")

    manifest2 = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest2, _FakeEmbedder())

    assert stats.added == 1
    assert stats.skipped == 1
    assert stats.removed == 1
    assert stats.updated == 0


# ---------------------------------------------------------------------------
# 8. Hidden-directory files (.obsidian/config.md) are skipped
# ---------------------------------------------------------------------------

def test_hidden_directory_files_are_skipped(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    hidden = vault / ".obsidian"
    hidden.mkdir()
    _write_md(hidden / "config.md", "Obsidian config", frontmatter="uuid: hidden-001")

    _write_md(vault / "real.md", "Real entry", frontmatter="uuid: real-001")

    store = _store()
    manifest = Manifest(tmp_path / "index-manifest.json")
    stats = run_index(vault, store, manifest, _FakeEmbedder())

    assert stats.added == 1
    assert stats.skipped == 0

    manifest.load()
    paths = manifest.all_paths()
    assert not any(".obsidian" in p for p in paths)
    assert "real.md" in paths
