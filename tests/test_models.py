from datetime import datetime, timezone
from jkb.models import DayOneEntry, NormalizedEntry, build_device_string


def test_dayone_entry_minimal():
    entry = DayOneEntry(
        uuid="ABC123",
        creationDate=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    assert entry.uuid == "ABC123"
    assert entry.text is None
    assert entry.tags == []
    assert entry.photos == []


def test_dayone_entry_extra_fields_allowed():
    entry = DayOneEntry(
        uuid="XYZ",
        creationDate=datetime(2020, 1, 1, tzinfo=timezone.utc),
        unknownFutureField="some value",
    )
    assert entry.uuid == "XYZ"


def test_normalized_entry_defaults():
    entry = NormalizedEntry(
        uuid="abc",
        journal="Philippines",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    assert entry.text == ""
    assert entry.tags == []
    assert entry.attachment_map == {}


def test_build_device_string_full():
    result = build_device_string("iPhone 13 Pro", "iOS", "16.1")
    assert result == "iPhone 13 Pro / iOS 16.1"


def test_build_device_string_device_only():
    result = build_device_string("MacBook Pro", None, None)
    assert result == "MacBook Pro"


def test_build_device_string_all_none():
    result = build_device_string(None, None, None)
    assert result is None


def test_build_device_string_os_only():
    result = build_device_string(None, "iOS", "16.1")
    assert result == "iOS 16.1"


# TC-MOD-05
def test_dayone_entry_allows_unknown_fields():
    from jkb.models.dayone import DayOneEntry
    entry = DayOneEntry(uuid="U1", creationDate="2020-01-01T00:00:00Z", futureField="value")
    assert entry.uuid == "U1"


# TC-MOD-06
def test_build_device_string_os_name_only():
    assert build_device_string(None, "macOS", None) == "macOS"


# TC-MOD-07
def test_normalized_entry_attachment_map_not_shared():
    from jkb.models.entry import NormalizedEntry
    from datetime import datetime, timezone
    e1 = NormalizedEntry(uuid="E1", journal="T", creation_date=datetime(2020,1,1,tzinfo=timezone.utc))
    e2 = NormalizedEntry(uuid="E2", journal="T", creation_date=datetime(2020,1,1,tzinfo=timezone.utc))
    e1.attachment_map["key"] = "val"
    assert "key" not in e2.attachment_map
