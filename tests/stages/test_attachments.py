import shutil
from datetime import datetime, timezone
from pathlib import Path

from jkb.models.dayone import DayOnePhoto, DayOneVideo, DayOneAudio, DayOnePDF
from jkb.models.entry import NormalizedEntry
from jkb.stages.attachments import handle_attachments


def _entry(**kwargs) -> NormalizedEntry:
    defaults = dict(
        uuid="abc12345",
        journal="Philippines",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return NormalizedEntry(**defaults)


def _setup_staging_photo(staging_dir: Path, journal: str, md5: str, ext: str = "jpeg") -> Path:
    src = staging_dir / journal / "photos" / f"{md5}.{ext}"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"fake_image")
    return src


def test_photo_copied_and_mapped(tmp_path):
    staging = tmp_path / "staging"
    output_dir = tmp_path / "output"
    _setup_staging_photo(staging, "Philippines", "abc123")

    photo = DayOnePhoto(identifier="PHOTO_ID_1", md5="abc123", type="jpeg")
    entry = _entry(photos=[photo])
    result = handle_attachments(entry, staging, output_dir)

    assert result.attachment_map["PHOTO_ID_1"] == "./attachments/abc123.jpeg"
    assert (output_dir / "attachments" / "abc123.jpeg").exists()


def test_missing_photo_silently_skipped(tmp_path):
    staging = tmp_path / "staging"
    output_dir = tmp_path / "output"

    photo = DayOnePhoto(identifier="MISSING_ID", md5="deadbeef", type="jpeg")
    entry = _entry(photos=[photo])
    result = handle_attachments(entry, staging, output_dir)

    # Map entry still added even for missing file
    assert "MISSING_ID" in result.attachment_map
    assert not (output_dir / "attachments" / "deadbeef.jpeg").exists()


def test_existing_dest_not_overwritten(tmp_path):
    staging = tmp_path / "staging"
    output_dir = tmp_path / "output"
    _setup_staging_photo(staging, "Philippines", "abc123")

    # Pre-create destination file with different content
    dest = output_dir / "attachments" / "abc123.jpeg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"original_content")

    photo = DayOnePhoto(identifier="PHOTO_ID_1", md5="abc123", type="jpeg")
    entry = _entry(photos=[photo])
    handle_attachments(entry, staging, output_dir)

    assert dest.read_bytes() == b"original_content"


def test_returns_new_entry_with_map(tmp_path):
    staging = tmp_path / "staging"
    output_dir = tmp_path / "output"

    photo = DayOnePhoto(identifier="P1", md5=None)  # no md5 → skip
    entry = _entry(photos=[photo])
    result = handle_attachments(entry, staging, output_dir)

    assert result is not entry  # model_copy returns new object
    assert result.attachment_map == {}


def test_multiple_media_types(tmp_path):
    staging = tmp_path / "staging"
    output_dir = tmp_path / "output"

    # Create a video in staging
    vid_src = staging / "Philippines" / "videos" / "vid001.mp4"
    vid_src.parent.mkdir(parents=True, exist_ok=True)
    vid_src.write_bytes(b"fake_video")

    video = DayOneVideo(identifier="VID_ID", md5="vid001", type="mp4")
    entry = _entry(videos=[video])
    result = handle_attachments(entry, staging, output_dir)

    assert result.attachment_map["VID_ID"] == "./attachments/vid001.mp4"
    assert (output_dir / "attachments" / "vid001.mp4").exists()
