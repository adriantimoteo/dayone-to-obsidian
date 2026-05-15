import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jkb.models.entry import NormalizedEntry
from jkb.stages.validate import validate, ValidationWarning


def _entry(uuid: str = "UUID1", journal: str = "Test", text: str = "Hello", **kwargs) -> NormalizedEntry:
    return NormalizedEntry(
        uuid=uuid,
        journal=journal,
        text=text,
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        **kwargs,
    )


def test_valid_entry_passes_through(tmp_path):
    entry = _entry()
    results = list(validate([entry], tmp_path))
    assert len(results) == 1
    entry_out, result = results[0]
    assert result.is_valid is True
    assert result.uuid == "UUID1"


def test_every_entry_yielded(tmp_path):
    entries = [_entry(uuid=f"UUID{i}") for i in range(5)]
    results = list(validate(entries, tmp_path))
    assert len(results) == 5


def test_duplicate_uuid_sets_invalid(tmp_path):
    e1 = _entry(uuid="SAME", journal="JournalA")
    e2 = _entry(uuid="SAME", journal="JournalB")
    results = list(validate([e1, e2], tmp_path))
    assert results[0][1].is_valid is True
    assert results[1][1].is_valid is False
    assert ValidationWarning.DUPLICATE_UUID in results[1][1].warnings
    assert results[1][1].duplicate_of_journal == "JournalA"


def test_missing_text_warning(tmp_path):
    entry = _entry(text="")
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.MISSING_TEXT in result.warnings
    assert result.is_valid is True


def test_media_only_warning(tmp_path):
    from jkb.models.dayone import DayOnePhoto
    photo = DayOnePhoto(identifier="PHOTO1", md5=None)
    entry = _entry(text="", photos=[photo])
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.MEDIA_ONLY in result.warnings
    assert ValidationWarning.MISSING_TEXT not in result.warnings


def test_missing_attachment_warning(tmp_path):
    from jkb.models.dayone import DayOnePhoto
    photo = DayOnePhoto(identifier="PHOTO1", md5="abc123", type="jpeg")
    entry = _entry(photos=[photo])
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.MISSING_ATTACHMENT in result.warnings
    assert "PHOTO1" in result.missing_attachment_ids


def test_attachment_present_no_warning(tmp_path):
    from jkb.models.dayone import DayOnePhoto
    photo_dir = tmp_path / "Test" / "photos"
    photo_dir.mkdir(parents=True)
    (photo_dir / "abc123.jpeg").write_bytes(b"img")
    photo = DayOnePhoto(identifier="PHOTO1", md5="abc123", type="jpeg")
    entry = _entry(photos=[photo])
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.MISSING_ATTACHMENT not in result.warnings


def test_sparse_metadata_warning(tmp_path):
    # Missing location, weather, activity, device, UTC timezone = 5 missing → sparse
    entry = _entry(location=None, weather=None, activity=None, device=None, timezone="UTC")
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.SPARSE_METADATA in result.warnings


def test_not_sparse_when_enough_metadata(tmp_path):
    from jkb.models.dayone import DayOneLocation, DayOneWeather
    loc = DayOneLocation(latitude=14.5, longitude=121.0)
    weather = DayOneWeather(temperatureCelsius=31.0)
    # Only missing activity, device (2 missing, timezone non-UTC) → below sparse threshold
    entry = _entry(location=loc, weather=weather, activity=None, device=None, timezone="Asia/Manila")
    _, result = list(validate([entry], tmp_path))[0]
    assert ValidationWarning.SPARSE_METADATA not in result.warnings


def test_uuid_dedup_scoped_to_call(tmp_path):
    """Two separate validate() calls each have their own seen set."""
    e = _entry(uuid="SAME")
    r1 = list(validate([e], tmp_path))
    r2 = list(validate([e], tmp_path))
    assert r1[0][1].is_valid is True
    assert r2[0][1].is_valid is True


# TC-VAL-08
def test_photo_with_no_md5_skips_attachment_check(tmp_path):
    from jkb.models.dayone import DayOnePhoto
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import validate, ValidationWarning
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="VAL08", journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        photos=[DayOnePhoto(identifier="P1", md5=None)],
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    pairs = list(validate([entry], staging))
    _, result = pairs[0]
    assert ValidationWarning.MISSING_ATTACHMENT not in result.warnings


# TC-VAL-09
def test_sparse_metadata_fires_once(tmp_path):
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import validate, ValidationWarning
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="VAL09", journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        timezone="UTC",  # no location, weather, activity, device, and UTC timezone
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    _, result = list(validate([entry], staging))[0]
    assert result.warnings.count(ValidationWarning.SPARSE_METADATA) == 1


# TC-VAL-10
def test_media_only_and_missing_attachment_coexist(tmp_path):
    from jkb.models.dayone import DayOnePhoto
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import validate, ValidationWarning
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="VAL10", journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="",
        photos=[DayOnePhoto(identifier="P1", md5="abc123")],
        # photo file absent from staging
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    _, result = list(validate([entry], staging))[0]
    assert ValidationWarning.MEDIA_ONLY in result.warnings
    assert ValidationWarning.MISSING_ATTACHMENT in result.warnings


# TC-VAL-11
def test_only_duplicate_uuid_sets_invalid(tmp_path):
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import validate
    from datetime import datetime, timezone
    # Entry with all warnings except DUPLICATE_UUID: no text, no location/weather/activity/device, UTC timezone
    entry = NormalizedEntry(
        uuid="VAL11", journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text="", timezone="UTC",
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    _, result = list(validate([entry], staging))[0]
    assert result.is_valid is True


# TC-VAL-12
def test_duplicate_of_journal_is_first_seen(tmp_path):
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import validate
    from datetime import datetime, timezone
    def _e(uuid, journal):
        return NormalizedEntry(uuid=uuid, journal=journal, creation_date=datetime(2020,1,1,tzinfo=timezone.utc))
    entries = [_e("X", "Alpha"), _e("X", "Beta"), _e("X", "Gamma")]
    staging = tmp_path / "staging"
    staging.mkdir()
    results = {r.journal: r for _, r in validate(entries, staging)}
    assert results["Beta"].duplicate_of_journal == "Alpha"
    assert results["Gamma"].duplicate_of_journal == "Alpha"


# TC-VAL-13
import pytest

@pytest.mark.parametrize("missing_count,expect_sparse", [
    (2, False),  # 2 missing → not sparse
    (3, True),   # 3 missing → sparse
    (4, True),   # 4 missing → sparse
])
def test_sparse_metadata_threshold(tmp_path, missing_count, expect_sparse):
    from jkb.models.dayone import DayOneLocation, DayOneWeather, DayOneActivity
    from jkb.models.entry import NormalizedEntry, build_device_string
    from jkb.stages.validate import validate, ValidationWarning
    from datetime import datetime, timezone

    # Fields that can be absent: location, weather, activity, device, timezone=="UTC"
    # missing_count controls how many are absent
    kwargs = dict(
        uuid="VAL13", journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    all_optional = ["location", "weather", "activity", "device", "utc_tz"]
    for i, field in enumerate(all_optional):
        if i < missing_count:
            pass  # leave absent
        else:
            if field == "location":
                kwargs["location"] = DayOneLocation(latitude=1.0, longitude=1.0)
            elif field == "weather":
                kwargs["weather"] = DayOneWeather(temperatureCelsius=20.0)
            elif field == "activity":
                kwargs["activity"] = DayOneActivity(stepCount=1000)
            elif field == "device":
                kwargs["device"] = "iPhone"
            elif field == "utc_tz":
                kwargs["timezone"] = "Asia/Manila"

    entry = NormalizedEntry(**kwargs)
    staging = tmp_path / "staging"
    staging.mkdir()
    _, result = list(validate([entry], staging))[0]
    sparse = ValidationWarning.SPARSE_METADATA in result.warnings
    assert sparse == expect_sparse
