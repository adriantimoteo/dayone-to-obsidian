from datetime import datetime, timezone
from pathlib import Path

import pytest

from jkb.models.entry import NormalizedEntry
from jkb.stages.write import write_entry, _filename


def _entry(**kwargs) -> NormalizedEntry:
    defaults = dict(
        uuid="A1B2C3D4e5f6",
        journal="Philippines",
        creation_date=datetime(2018, 4, 12, 6, 32, 0, tzinfo=timezone.utc),
        timezone="Asia/Manila",
    )
    defaults.update(kwargs)
    return NormalizedEntry(**defaults)


def _frontmatter() -> dict:
    return {
        "date": "2018-04-12T06:32:00Z",
        "journal": "Philippines",
        "starred": False,
        "location": {"coords": None, "name": None, "country": None, "label": None},
        "weather": {"temp_c": None, "condition": None, "wind_kph": None, "humidity": None},
        "activity": {"steps": None, "motion": None},
        "device": None,
        "timezone": "Asia/Manila",
        "tags": [],
        "people": [],
        "pinned": False,
        "duration_seconds": None,
        "modified": None,
    }


def test_file_created_in_correct_directory(tmp_path):
    entry = _entry()
    path = write_entry(entry, _frontmatter(), "Hello world\n", tmp_path)
    assert path is not None
    # Asia/Manila is UTC+8, so 06:32 UTC = 14:32 Manila
    assert path.parent.parent.name == "2018"
    assert path.parent.name == "04"


def test_filename_format(tmp_path):
    entry = _entry()
    path = write_entry(entry, _frontmatter(), "Hello\n", tmp_path)
    assert path is not None
    # UTC 06:32 → Manila 14:32
    assert path.name == "2018-04-12-1432-a1b2c3d4.md"


def test_file_content_has_frontmatter_and_body(tmp_path):
    entry = _entry()
    path = write_entry(entry, _frontmatter(), "Hello world\n", tmp_path)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "---\n\n" in content
    assert "Hello world" in content


def test_yaml_date_is_string_not_timestamp(tmp_path):
    entry = _entry()
    fm = _frontmatter()
    fm["date"] = "2018-04-12T06:32:00Z"
    path = write_entry(entry, fm, "body\n", tmp_path)
    content = path.read_text(encoding="utf-8")
    assert "2018-04-12T06:32:00Z" in content
    # ruamel.yaml must not render it as a datetime object
    assert "2018-04-12 " not in content


def test_skip_if_exists_no_overwrite(tmp_path):
    entry = _entry()
    path1 = write_entry(entry, _frontmatter(), "First\n", tmp_path)
    path2 = write_entry(entry, _frontmatter(), "Second\n", tmp_path, overwrite=False)
    assert path2 is None
    assert path1.read_text(encoding="utf-8").count("First") == 1


def test_overwrite_replaces_file(tmp_path):
    entry = _entry()
    write_entry(entry, _frontmatter(), "First\n", tmp_path)
    write_entry(entry, _frontmatter(), "Second\n", tmp_path, overwrite=True)
    # Re-derive the path
    from jkb.stages.write import _filename
    from zoneinfo import ZoneInfo
    local_dt = entry.creation_date.astimezone(ZoneInfo("Asia/Manila"))
    p = tmp_path / local_dt.strftime("%Y") / local_dt.strftime("%m") / _filename(entry)
    assert "Second" in p.read_text(encoding="utf-8")


def test_utc_fallback_for_unknown_timezone(tmp_path):
    entry = _entry(timezone="Unknown/Zone")
    path = write_entry(entry, _frontmatter(), "body\n", tmp_path)
    assert path is not None
    # Falls back to UTC: 06:32 UTC → directory 2018/04 still correct
    assert path.parent.parent.name == "2018"
    assert path.parent.name == "04"
