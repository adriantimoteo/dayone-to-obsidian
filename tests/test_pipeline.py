import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jkb.models.entry import NormalizedEntry
from jkb.stages.validate import ValidationResult, ValidationWarning
from jkb.utils.checkpoint import Checkpoint, STAGE_WRITE
from jkb.utils.migration_log import MigrationLog
from jkb.pipeline import process_entry


def _entry(uuid: str = "ABCD1234efgh", **kwargs) -> NormalizedEntry:
    defaults = dict(
        uuid=uuid,
        journal="Test",
        creation_date=datetime(2020, 6, 15, 10, 30, tzinfo=timezone.utc),
        timezone="UTC",
    )
    defaults.update(kwargs)
    return NormalizedEntry(**defaults)


def _valid_result(uuid: str = "ABCD1234efgh") -> ValidationResult:
    return ValidationResult(uuid=uuid, journal="Test", is_valid=True)


def _invalid_result(uuid: str = "ABCD1234efgh") -> ValidationResult:
    return ValidationResult(
        uuid=uuid, journal="Test", is_valid=False,
        warnings=[ValidationWarning.DUPLICATE_UUID],
        duplicate_of_journal="Other",
    )


def test_valid_entry_written(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = Checkpoint(tmp_path / "cp.json")
    log = MigrationLog()

    process_entry(_entry(), _valid_result(), staging, output, checkpoint, log, overwrite=False)

    assert log._total_written == 1
    assert checkpoint.is_done(STAGE_WRITE, "ABCD1234efgh")


def test_invalid_entry_skipped(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = Checkpoint(tmp_path / "cp.json")
    log = MigrationLog()

    process_entry(_entry(), _invalid_result(), staging, output, checkpoint, log, overwrite=False)

    assert log._total_written == 0
    assert log._total_skipped == 1
    assert not checkpoint.is_done(STAGE_WRITE, "ABCD1234efgh")


def test_already_checkpointed_skipped(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = Checkpoint(tmp_path / "cp.json")
    checkpoint.mark_done(STAGE_WRITE, "ABCD1234efgh")
    log = MigrationLog()

    process_entry(_entry(), _valid_result(), staging, output, checkpoint, log, overwrite=False)

    # Skipped (already done), not written again
    assert log._total_written == 0
    assert log._total_skipped == 1


def test_entry_not_double_processed(tmp_path):
    """Running process_entry twice for the same entry with no overwrite → only written once."""
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = Checkpoint(tmp_path / "cp.json")
    log = MigrationLog()

    process_entry(_entry(), _valid_result(), staging, output, checkpoint, log, overwrite=False)
    process_entry(_entry(), _valid_result(), staging, output, checkpoint, log, overwrite=False)

    assert log._total_written == 1
    assert log._total_skipped == 1
