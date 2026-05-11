from __future__ import annotations
import logging
from pathlib import Path

from jkb.models.entry import NormalizedEntry
from jkb.stages.attachments import handle_attachments
from jkb.stages.body import build_body
from jkb.stages.frontmatter import build_frontmatter
from jkb.stages.validate import ValidationResult
from jkb.stages.write import _local_datetime, write_entry
from jkb.utils.checkpoint import Checkpoint, STAGE_WRITE
from jkb.utils.migration_log import MigrationLog

logger = logging.getLogger(__name__)


def process_entry(
    entry: NormalizedEntry,
    result: ValidationResult,
    staging_dir: Path,
    output_root: Path,
    checkpoint: Checkpoint,
    log: MigrationLog,
    overwrite: bool,
) -> None:
    # 1. Skip invalid entries
    if not result.is_valid:
        log.record(entry, result, written_path=None)
        return

    # 2. Skip already-done entries
    if checkpoint.is_done(STAGE_WRITE, entry.uuid):
        log.record(entry, result, written_path=None)
        return

    # 3. Resolve target directory using local timezone (matches write_entry path)
    local_dt = _local_datetime(entry)
    entry_dir = output_root / local_dt.strftime("%Y") / local_dt.strftime("%m")

    # 4. Handle attachments (populates entry.attachment_map)
    entry = handle_attachments(entry, staging_dir, entry_dir)

    # 5. Build frontmatter and body
    frontmatter = build_frontmatter(entry)
    body = build_body(entry)

    # 6. Write
    try:
        written_path = write_entry(entry, frontmatter, body, output_root, overwrite)
    except Exception as e:
        logger.error("Failed to write entry %s: %s", entry.uuid, e)
        checkpoint.mark_failed(STAGE_WRITE, entry.uuid, str(e))
        log.record(entry, result, written_path=None)
        return

    # 7. Mark checkpoint and record log
    if written_path is not None:
        checkpoint.mark_done(STAGE_WRITE, entry.uuid)
    log.record(entry, result, written_path)
