from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path) -> None:
        self._path = path
        # Each entry: {"hash": str, "entry_id": str}
        self._data: dict[str, dict[str, str]] = {}

    def load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        # Support both old format (plain string) and new format (dict)
        for k, v in raw.items():
            if isinstance(v, str):
                self._data[k] = {"hash": v, "entry_id": k}
            else:
                self._data[k] = v

    def save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def get_hash(self, rel_path: str) -> str | None:
        entry = self._data.get(rel_path)
        return entry["hash"] if entry else None

    def get_entry_id(self, rel_path: str) -> str | None:
        entry = self._data.get(rel_path)
        return entry["entry_id"] if entry else None

    def set_hash(self, rel_path: str, hash_: str, entry_id: str | None = None) -> None:
        self._data[rel_path] = {
            "hash": hash_,
            "entry_id": entry_id if entry_id is not None else rel_path,
        }

    def remove(self, rel_path: str) -> None:
        self._data.pop(rel_path, None)

    def all_paths(self) -> set[str]:
        return set(self._data.keys())
