from __future__ import annotations
import json
import logging
import zipfile
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from jkb.models.dayone import DayOneEntry
from jkb.models.entry import NormalizedEntry, build_device_string

logger = logging.getLogger(__name__)

MEDIA_DIRS = ("photos", "videos", "audios", "pdfs")


def _extract_media(zf: zipfile.ZipFile, journal_name: str, staging_dir: Path) -> None:
    """Extract all media files from ZIP to staging_dir/{journal_name}/{media_type}/"""
    for name in zf.namelist():
        parts = Path(name).parts
        if len(parts) >= 2 and parts[0] in MEDIA_DIRS:
            media_type = parts[0]
            filename = parts[-1]
            dest_dir = staging_dir / journal_name / media_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            if not dest_path.exists():
                data = zf.read(name)
                dest_path.write_bytes(data)


def _normalize(raw: DayOneEntry, journal_name: str) -> NormalizedEntry:
    """Convert a DayOneEntry to NormalizedEntry."""
    device = build_device_string(raw.deviceName, raw.creationOSName, raw.creationOSVersion)
    return NormalizedEntry(
        uuid=raw.uuid,
        journal=journal_name,
        text=raw.text or "",
        creation_date=raw.creationDate,
        modified_date=raw.modifiedDate,
        timezone=raw.timeZone or "UTC",
        starred=raw.starred,
        pinned=raw.pinned,
        tags=raw.tags,
        device=device,
        location=raw.location,
        weather=raw.weather,
        activity=raw.activity,
        photos=raw.photos,
        videos=raw.videos,
        audios=raw.audios,
        pdfs=raw.pdfAttachments,
        duration_seconds=raw.duration,
        source_string=raw.sourceString,
    )


def _parse_zip(zip_path: Path, staging_dir: Path) -> Iterator[NormalizedEntry]:
    """Parse a single .dayone ZIP file, yielding NormalizedEntry objects."""
    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning("Cannot open ZIP %s: %s", zip_path, e)
        return

    with zf:
        # Find all .json files in the ZIP (each is a journal)
        json_names = [n for n in zf.namelist() if n.endswith(".json") and "/" not in n]
        if not json_names:
            logger.warning("No journal JSON found in %s", zip_path)
            return

        for json_name in json_names:
            journal_name = Path(json_name).stem

            try:
                raw_bytes = zf.read(json_name)
                data = json.loads(raw_bytes)
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning("Cannot parse JSON %s in %s: %s", json_name, zip_path, e)
                continue

            # Extract media for this journal
            _extract_media(zf, journal_name, staging_dir)

            entries_data = data.get("entries", [])
            for entry_data in entries_data:
                try:
                    raw = DayOneEntry.model_validate(entry_data)
                    yield _normalize(raw, journal_name)
                except (ValidationError, Exception) as e:
                    uuid = entry_data.get("uuid", "<unknown>")
                    logger.warning("Skipping corrupt entry %s in %s/%s: %s", uuid, zip_path.name, journal_name, e)


def parse(input_path: Path, staging_dir: Path) -> Iterator[NormalizedEntry]:
    """
    Yields NormalizedEntry for every entry found in input_path.
    input_path may be:
      - a single .dayone ZIP file
      - a directory containing .dayone ZIP files (non-recursive)
    Corrupt entries are logged as warnings and skipped.
    Media files are extracted to staging_dir/{journal_name}/{media_type}/
    """
    if input_path.is_file():
        yield from _parse_zip(input_path, staging_dir)
    elif input_path.is_dir():
        zip_files = sorted(input_path.glob("*.dayone"))
        if not zip_files:
            logger.warning("No .dayone files found in directory %s", input_path)
        for zip_path in zip_files:
            yield from _parse_zip(zip_path, staging_dir)
    else:
        logger.warning("Input path does not exist: %s", input_path)
