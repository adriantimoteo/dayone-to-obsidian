from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel

from jkb.models.entry import NormalizedEntry

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = {
    "photos": "jpeg",
    "videos": "mp4",
    "audios": "m4a",
    "pdfs": "pdf",
}


class ValidationWarning(str, Enum):
    MISSING_TEXT = "missing_text"
    MISSING_LOCATION = "missing_location"
    MISSING_WEATHER = "missing_weather"
    MISSING_ACTIVITY = "missing_activity"
    MISSING_ATTACHMENT = "missing_attachment"
    DUPLICATE_UUID = "duplicate_uuid"
    SPARSE_METADATA = "sparse_metadata"
    MEDIA_ONLY = "media_only"


class ValidationResult(BaseModel):
    uuid: str
    journal: str
    is_valid: bool = True
    warnings: list[ValidationWarning] = []
    missing_attachment_ids: list[str] = []
    duplicate_of_journal: str | None = None


def _check_attachments(
    entry: NormalizedEntry,
    staging_dir: Path,
    result: ValidationResult,
) -> None:
    """Check that referenced media files exist in staging_dir."""
    # photos
    for photo in entry.photos:
        if photo.md5:
            ext = photo.type or "jpeg"
            path = staging_dir / entry.journal / "photos" / f"{photo.md5}.{ext}"
            if not path.exists():
                result.warnings.append(ValidationWarning.MISSING_ATTACHMENT)
                result.missing_attachment_ids.append(photo.identifier)
    # videos
    for video in entry.videos:
        if video.md5:
            ext = video.type or "mp4"
            path = staging_dir / entry.journal / "videos" / f"{video.md5}.{ext}"
            if not path.exists():
                result.warnings.append(ValidationWarning.MISSING_ATTACHMENT)
                result.missing_attachment_ids.append(video.identifier)
    # audios
    for audio in entry.audios:
        if audio.md5:
            ext = audio.type or "m4a"
            path = staging_dir / entry.journal / "audios" / f"{audio.md5}.{ext}"
            if not path.exists():
                result.warnings.append(ValidationWarning.MISSING_ATTACHMENT)
                result.missing_attachment_ids.append(audio.identifier)
    # pdfs
    for pdf in entry.pdfs:
        if pdf.md5:
            ext = pdf.type or "pdf"
            path = staging_dir / entry.journal / "pdfs" / f"{pdf.md5}.{ext}"
            if not path.exists():
                result.warnings.append(ValidationWarning.MISSING_ATTACHMENT)
                result.missing_attachment_ids.append(pdf.identifier)


def _check_sparse_metadata(entry: NormalizedEntry, result: ValidationResult) -> None:
    """Flag entries missing 3+ of: location, weather, activity, device, non-UTC timezone."""
    missing_count = sum([
        entry.location is None,
        entry.weather is None,
        entry.activity is None,
        entry.device is None,
        entry.timezone == "UTC",
    ])
    if missing_count >= 4:
        result.warnings.append(ValidationWarning.SPARSE_METADATA)


def validate(
    entries: Iterable[NormalizedEntry],
    staging_dir: Path,
) -> Iterator[tuple[NormalizedEntry, ValidationResult]]:
    """
    Yields (entry, result) for every input entry. Never drops entries.
    is_valid=False ONLY for DUPLICATE_UUID.
    All other warnings are informational.
    UUID deduplication is scoped to this validate() call.
    """
    seen: dict[str, str] = {}  # uuid → journal

    for entry in entries:
        result = ValidationResult(uuid=entry.uuid, journal=entry.journal)

        # 1. Duplicate UUID check
        if entry.uuid in seen:
            result.is_valid = False
            result.warnings.append(ValidationWarning.DUPLICATE_UUID)
            result.duplicate_of_journal = seen[entry.uuid]
            yield entry, result
            continue

        seen[entry.uuid] = entry.journal

        # 2. Text / media checks
        has_text = bool(entry.text.strip())
        has_media = bool(entry.photos or entry.videos or entry.audios or entry.pdfs)

        if not has_text and has_media:
            result.warnings.append(ValidationWarning.MEDIA_ONLY)
        elif not has_text:
            result.warnings.append(ValidationWarning.MISSING_TEXT)

        # 3. Optional metadata warnings
        if entry.location is None:
            result.warnings.append(ValidationWarning.MISSING_LOCATION)
        if entry.weather is None:
            result.warnings.append(ValidationWarning.MISSING_WEATHER)
        if entry.activity is None:
            result.warnings.append(ValidationWarning.MISSING_ACTIVITY)

        # 4. Sparse metadata check
        _check_sparse_metadata(entry, result)

        # 5. Attachment existence
        _check_attachments(entry, staging_dir, result)

        yield entry, result
