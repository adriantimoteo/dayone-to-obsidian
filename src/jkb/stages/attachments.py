from __future__ import annotations
import logging
import shutil
from pathlib import Path

from jkb.models.entry import NormalizedEntry

logger = logging.getLogger(__name__)

_MEDIA_DIRS = {
    "photos": "photos",
    "videos": "videos",
    "audios": "audios",
    "pdfs": "pdfs",
}


def _copy_file(src: Path, dst: Path) -> bool:
    """Copy src to dst. Returns True if copied, False if skipped or missing."""
    if not src.exists():
        return False
    if dst.exists():
        return True  # already there from a previous run
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def handle_attachments(
    entry: NormalizedEntry,
    staging_dir: Path,
    entry_output_dir: Path,
) -> NormalizedEntry:
    """
    Copies media files from staging_dir to entry_output_dir/attachments/.
    Returns entry with attachment_map populated.
    Missing files silently skipped (already flagged by validate stage).
    """
    attachment_map: dict[str, str] = {}
    attachments_dir = entry_output_dir / "attachments"

    # Photos
    for photo in entry.photos:
        if photo.md5:
            ext = photo.type or "jpeg"
            src = staging_dir / entry.journal / "photos" / f"{photo.md5}.{ext}"
            dst = attachments_dir / f"{photo.md5}.{ext}"
            _copy_file(src, dst)
            attachment_map[photo.identifier] = f"./attachments/{photo.md5}.{ext}"

    # Videos
    for video in entry.videos:
        if video.md5:
            ext = video.type or "mp4"
            src = staging_dir / entry.journal / "videos" / f"{video.md5}.{ext}"
            dst = attachments_dir / f"{video.md5}.{ext}"
            _copy_file(src, dst)
            attachment_map[video.identifier] = f"./attachments/{video.md5}.{ext}"

    # Audios
    for audio in entry.audios:
        if audio.md5:
            ext = audio.type or "m4a"
            src = staging_dir / entry.journal / "audios" / f"{audio.md5}.{ext}"
            dst = attachments_dir / f"{audio.md5}.{ext}"
            _copy_file(src, dst)
            attachment_map[audio.identifier] = f"./attachments/{audio.md5}.{ext}"

    # PDFs
    for pdf in entry.pdfs:
        if pdf.md5:
            ext = pdf.type or "pdf"
            src = staging_dir / entry.journal / "pdfs" / f"{pdf.md5}.{ext}"
            dst = attachments_dir / f"{pdf.md5}.{ext}"
            _copy_file(src, dst)
            attachment_map[pdf.identifier] = f"./attachments/{pdf.md5}.{ext}"

    return entry.model_copy(update={"attachment_map": attachment_map})
