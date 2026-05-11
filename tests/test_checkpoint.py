import json
from pathlib import Path

import pytest

from jkb.utils.checkpoint import Checkpoint, STAGE_WRITE


def test_mark_done_and_is_done(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    assert not cp.is_done(STAGE_WRITE, "uuid-1")
    cp.mark_done(STAGE_WRITE, "uuid-1")
    assert cp.is_done(STAGE_WRITE, "uuid-1")


def test_persists_across_instances(tmp_path):
    path = tmp_path / "checkpoint.json"
    cp1 = Checkpoint(path)
    cp1.mark_done(STAGE_WRITE, "uuid-A")
    cp1.mark_done(STAGE_WRITE, "uuid-B")

    cp2 = Checkpoint(path)
    assert cp2.is_done(STAGE_WRITE, "uuid-A")
    assert cp2.is_done(STAGE_WRITE, "uuid-B")
    assert not cp2.is_done(STAGE_WRITE, "uuid-C")


def test_mark_failed_and_is_failed(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_failed(STAGE_WRITE, "uuid-X", "some error")
    assert cp.is_failed(STAGE_WRITE, "uuid-X")
    assert not cp.is_failed(STAGE_WRITE, "uuid-Y")


def test_counts(tmp_path):
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_done(STAGE_WRITE, "a")
    cp.mark_done(STAGE_WRITE, "b")
    cp.mark_failed(STAGE_WRITE, "c", "error")
    counts = cp.counts(STAGE_WRITE)
    assert counts["done"] == 2
    assert counts["failed"] == 1


def test_clear_removes_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    cp = Checkpoint(path)
    cp.mark_done(STAGE_WRITE, "uuid-1")
    assert path.exists()
    cp.clear()
    assert not path.exists()
    assert not cp.is_done(STAGE_WRITE, "uuid-1")


def test_corrupt_file_starts_fresh(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text("not valid json", encoding="utf-8")
    cp = Checkpoint(path)
    assert not cp.is_done(STAGE_WRITE, "anything")


def test_atomic_write(tmp_path):
    """mark_done writes file immediately (write-through)."""
    path = tmp_path / "checkpoint.json"
    cp = Checkpoint(path)
    cp.mark_done(STAGE_WRITE, "uuid-1")
    assert path.exists()
    data = json.loads(path.read_text())
    assert "uuid-1" in data["stages"][STAGE_WRITE]["done"]


def test_idempotent_mark_done(tmp_path):
    """Calling mark_done twice for the same uuid is safe."""
    cp = Checkpoint(tmp_path / "checkpoint.json")
    cp.mark_done(STAGE_WRITE, "uuid-1")
    cp.mark_done(STAGE_WRITE, "uuid-1")
    assert cp.counts(STAGE_WRITE)["done"] == 1
