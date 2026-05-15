import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jkb.stages.parse import parse


def _make_dayone_zip(tmp_path: Path, journal_name: str, entries: list[dict], photos: dict[str, bytes] | None = None) -> Path:
    """Helper: create a minimal .dayone ZIP in tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_path / f"{journal_name}.dayone"
    with zipfile.ZipFile(zip_path, "w") as zf:
        payload = {"entries": entries}
        zf.writestr(f"{journal_name}.json", json.dumps(payload))
        if photos:
            for filename, data in photos.items():
                zf.writestr(f"photos/{filename}", data)
    return zip_path


def _minimal_entry(uuid: str = "ABC123DEF456") -> dict:
    return {
        "uuid": uuid,
        "creationDate": "2020-06-15T10:30:00Z",
        "text": "Hello world",
        "timeZone": "Asia/Manila",
        "starred": False,
        "pinned": False,
        "tags": [],
    }


def test_parse_single_zip(tmp_path):
    zip_path = _make_dayone_zip(tmp_path / "zips", "Philippines", [_minimal_entry()])
    staging = tmp_path / "staging"
    staging.mkdir()

    entries = list(parse(zip_path, staging))
    assert len(entries) == 1
    assert entries[0].uuid == "ABC123DEF456"
    assert entries[0].journal == "Philippines"
    assert entries[0].text == "Hello world"
    assert entries[0].timezone == "Asia/Manila"


def test_parse_directory_of_zips(tmp_path):
    zip_dir = tmp_path / "zips"
    zip_dir.mkdir()
    _make_dayone_zip(zip_dir, "Journal1", [_minimal_entry("UUID001")])
    _make_dayone_zip(zip_dir, "Journal2", [_minimal_entry("UUID002")])
    staging = tmp_path / "staging"
    staging.mkdir()

    entries = list(parse(zip_dir, staging))
    assert len(entries) == 2
    journals = {e.journal for e in entries}
    assert journals == {"Journal1", "Journal2"}


def test_corrupt_entry_skipped(tmp_path):
    good = _minimal_entry("GOOD1")
    bad = {"uuid": "BAD1"}  # missing required creationDate
    zip_path = _make_dayone_zip(tmp_path / "zips", "Test", [good, bad])
    staging = tmp_path / "staging"
    staging.mkdir()

    entries = list(parse(zip_path, staging))
    assert len(entries) == 1
    assert entries[0].uuid == "GOOD1"


def test_media_extracted_to_staging(tmp_path):
    entry = _minimal_entry()
    photos = {"abc123.jpeg": b"fake_image_data"}
    zip_path = _make_dayone_zip(tmp_path / "zips", "Philippines", [entry], photos=photos)
    staging = tmp_path / "staging"
    staging.mkdir()

    list(parse(zip_path, staging))
    assert (staging / "Philippines" / "photos" / "abc123.jpeg").exists()


def test_bad_zip_skipped(tmp_path):
    bad_zip = tmp_path / "bad.dayone"
    bad_zip.write_bytes(b"not a zip file")
    staging = tmp_path / "staging"
    staging.mkdir()

    entries = list(parse(bad_zip, staging))
    assert entries == []


def test_empty_directory(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(tmp_path, staging))
    assert entries == []


def test_journal_name_from_json_filename(tmp_path):
    zip_path = _make_dayone_zip(tmp_path / "zips", "MyDiary", [_minimal_entry()])
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert entries[0].journal == "MyDiary"


# TC-PARSE-08
def test_parse_zip_no_media_dirs(tmp_path):
    zip_path = _make_dayone_zip(tmp_path / "zips", "NoMedia", [_minimal_entry()])
    # No photos= kwarg → no media dirs in ZIP
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert len(entries) == 1
    assert not any(staging.rglob("*.*"))  # nothing extracted


# TC-PARSE-09
def test_parse_multi_journal_zip(tmp_path):
    zip_path = tmp_path / "Multi.dayone"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Journal1.json", json.dumps({"entries": [_minimal_entry("J1-001"), _minimal_entry("J1-002")]}))
        zf.writestr("Journal2.json", json.dumps({"entries": [_minimal_entry("J2-001"), _minimal_entry("J2-002")]}))
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert len(entries) == 4
    j1 = [e for e in entries if e.journal == "Journal1"]
    j2 = [e for e in entries if e.journal == "Journal2"]
    assert len(j1) == 2
    assert len(j2) == 2


# TC-PARSE-10
def test_parse_media_not_reextracted_if_present(tmp_path):
    photos = {"abc123.jpeg": b"original_bytes"}
    zip_path = _make_dayone_zip(tmp_path / "zips", "Test", [_minimal_entry()], photos=photos)
    staging = tmp_path / "staging"
    staging.mkdir()
    # First parse — extracts file
    list(parse(zip_path, staging))
    extracted = staging / "Test" / "photos" / "abc123.jpeg"
    assert extracted.exists()
    # Overwrite with sentinel
    extracted.write_bytes(b"sentinel")
    # Second parse — should NOT re-extract
    list(parse(zip_path, staging))
    assert extracted.read_bytes() == b"sentinel"


# TC-PARSE-11
def test_parse_missing_timezone_defaults_to_utc(tmp_path):
    entry = _minimal_entry()
    del entry["timeZone"]
    zip_path = _make_dayone_zip(tmp_path / "zips", "Test", [entry])
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert entries[0].timezone == "UTC"


# TC-PARSE-12
def test_parse_null_text_normalized(tmp_path):
    entry = _minimal_entry()
    entry["text"] = None
    zip_path = _make_dayone_zip(tmp_path / "zips", "Test", [entry])
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert entries[0].text == ""


# TC-PARSE-13
def test_parse_photos_and_pdfs(tmp_path):
    entry = _minimal_entry()
    entry["photos"] = [{"identifier": "P1", "md5": "aaa", "type": "jpeg"}]
    entry["pdfAttachments"] = [{"identifier": "D1", "md5": "bbb", "type": "pdf"}]
    zip_path = _make_dayone_zip(tmp_path / "zips", "Test", [entry])
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_path, staging))
    assert len(entries[0].photos) == 1
    assert len(entries[0].pdfs) == 1


# TC-PARSE-14
def test_parse_nonexistent_path(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(tmp_path / "nonexistent.dayone", staging))
    assert entries == []


# TC-PARSE-15
def test_parse_directory_ignores_non_dayone(tmp_path):
    zip_dir = tmp_path / "zips"
    _make_dayone_zip(zip_dir, "Journal", [_minimal_entry()])
    (zip_dir / "notes.txt").write_text("ignore me")
    staging = tmp_path / "staging"
    staging.mkdir()
    entries = list(parse(zip_dir, staging))
    assert len(entries) == 1
