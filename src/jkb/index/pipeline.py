from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML

from jkb.index.chunker import chunk_entry
from jkb.index.embedder import Embedder, embed_chunks
from jkb.index.manifest import Manifest, md5_file
from jkb.index.vectorstore import VectorStore

_yaml = YAML()


def _sanitize_metadata(raw: dict) -> dict:
    """Convert non-primitive YAML values to strings so ChromaDB can store them."""
    out = {}
    for k, v in raw.items():
        if v is None or isinstance(v, (str, int, float, bool, list)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _parse_md(file_path: Path) -> tuple[dict, str]:
    text = file_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return {}, text
    frontmatter_text = rest[:end]
    body = rest[end + 4:]
    if body.startswith("\n"):
        body = body[1:]
    raw = dict(_yaml.load(StringIO(frontmatter_text)) or {})
    return _sanitize_metadata(raw), body


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0


def run_index(
    vault_path: Path,
    store: VectorStore,
    manifest: Manifest,
    embedder: Embedder,
    *,
    force: bool = False,
) -> IndexStats:
    stats = IndexStats()

    manifest.load()

    current_files: dict[str, Path] = {}
    for md_file in vault_path.rglob("*.md"):
        rel = md_file.relative_to(vault_path)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        current_files[rel.as_posix()] = md_file

    for rel_path, abs_path in current_files.items():
        file_hash = md5_file(abs_path)
        existing_hash = manifest.get_hash(rel_path)

        if not force and existing_hash == file_hash:
            stats.skipped += 1
            continue

        is_new = existing_hash is None
        metadata, body = _parse_md(abs_path)
        entry_id = str(metadata.get("uuid", rel_path))
        chunks = chunk_entry(body, {**metadata, "uuid": entry_id})
        chunks_and_embeddings = embed_chunks(chunks, embedder)
        store.delete(entry_id)
        store.add(chunks_and_embeddings)
        manifest.set_hash(rel_path, file_hash, entry_id=entry_id)

        if is_new:
            stats.added += 1
        else:
            stats.updated += 1

    manifest_paths = manifest.all_paths()
    removed_paths = manifest_paths - set(current_files.keys())
    for rel_path in removed_paths:
        entry_id = manifest.get_entry_id(rel_path) or rel_path
        store.delete(entry_id)
        manifest.remove(rel_path)
        stats.removed += 1

    manifest.save()
    return stats
