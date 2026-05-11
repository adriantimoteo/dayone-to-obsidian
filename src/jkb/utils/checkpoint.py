from __future__ import annotations
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE_WRITE = "write"

_SCHEMA_VERSION = 1


def _empty_data() -> dict:
    return {"version": _SCHEMA_VERSION, "stages": {}}


def _empty_stage() -> dict:
    return {"done": [], "failed": {}}


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict = _empty_data()
        self._done_sets: dict[str, set[str]] = {}

        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("version") == _SCHEMA_VERSION:
                    self._data = raw
                else:
                    logger.warning("Checkpoint version mismatch; starting fresh: %s", path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Corrupt checkpoint, starting fresh (%s): %s", path, e)

        # Build in-memory sets for O(1) lookups
        for stage, stage_data in self._data["stages"].items():
            self._done_sets[stage] = set(stage_data.get("done", []))

    def _ensure_stage(self, stage: str) -> None:
        if stage not in self._data["stages"]:
            self._data["stages"][stage] = _empty_stage()
        if stage not in self._done_sets:
            self._done_sets[stage] = set()

    def _save(self) -> None:
        """Atomically write checkpoint to disk."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent,
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def is_done(self, stage: str, uuid: str) -> bool:
        return uuid in self._done_sets.get(stage, set())

    def mark_done(self, stage: str, uuid: str) -> None:
        self._ensure_stage(stage)
        if uuid not in self._done_sets[stage]:
            self._done_sets[stage].add(uuid)
            self._data["stages"][stage]["done"].append(uuid)
            self._save()

    def mark_failed(self, stage: str, uuid: str, reason: str) -> None:
        self._ensure_stage(stage)
        self._data["stages"][stage]["failed"][uuid] = reason
        self._save()

    def is_failed(self, stage: str, uuid: str) -> bool:
        stage_data = self._data["stages"].get(stage, {})
        return uuid in stage_data.get("failed", {})

    def counts(self, stage: str) -> dict[str, int]:
        stage_data = self._data["stages"].get(stage, _empty_stage())
        return {
            "done": len(stage_data["done"]),
            "failed": len(stage_data["failed"]),
        }

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
        self._data = _empty_data()
        self._done_sets = {}
