from datetime import datetime, timezone
from jkb.models.entry import NormalizedEntry
from jkb.models.dayone import DayOneLocation, DayOneWeather, DayOneActivity
from jkb.stages.frontmatter import build_frontmatter


def _entry(**kwargs) -> NormalizedEntry:
    defaults = dict(
        uuid="abc12345",
        journal="Philippines",
        creation_date=datetime(2018, 4, 12, 6, 32, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return NormalizedEntry(**defaults)


def test_all_keys_present():
    fm = build_frontmatter(_entry())
    for key in ["date", "journal", "starred", "location", "weather", "activity",
                "device", "timezone", "tags", "people", "pinned", "duration_seconds", "modified"]:
        assert key in fm, f"Missing key: {key}"


def test_date_format():
    fm = build_frontmatter(_entry())
    assert fm["date"] == "2018-04-12T06:32:00Z"


def test_modified_none():
    fm = build_frontmatter(_entry())
    assert fm["modified"] is None


def test_modified_present():
    mod = datetime(2018, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    fm = build_frontmatter(_entry(modified_date=mod))
    assert fm["modified"] == "2018-04-13T12:00:00Z"


def test_location_with_coords():
    loc = DayOneLocation(latitude=14.5547, longitude=121.0244, placeName="Poblacion", localityName="Makati", country="Philippines")
    fm = build_frontmatter(_entry(location=loc))
    assert fm["location"]["coords"] == [14.5547, 121.0244]
    assert fm["location"]["name"] == "Poblacion, Makati"
    assert fm["location"]["country"] == "Philippines"


def test_location_none():
    fm = build_frontmatter(_entry(location=None))
    assert fm["location"]["coords"] is None
    assert fm["location"]["name"] is None
    assert fm["location"]["country"] is None
    assert fm["location"]["label"] is None


def test_weather_present():
    wx = DayOneWeather(temperatureCelsius=31.0, conditionDescription="Sunny")
    fm = build_frontmatter(_entry(weather=wx))
    assert fm["weather"]["temp_c"] == 31.0
    assert fm["weather"]["condition"] == "Sunny"


def test_weather_none():
    fm = build_frontmatter(_entry(weather=None))
    assert fm["weather"]["temp_c"] is None
    assert fm["weather"]["condition"] is None


def test_activity_present():
    act = DayOneActivity(stepCount=4200, activityName="walking")
    fm = build_frontmatter(_entry(activity=act))
    assert fm["activity"]["steps"] == 4200
    assert fm["activity"]["motion"] == "walking"


def test_activity_none():
    fm = build_frontmatter(_entry(activity=None))
    assert fm["activity"]["steps"] is None
    assert fm["activity"]["motion"] is None


def test_people_always_empty_list():
    fm = build_frontmatter(_entry())
    assert fm["people"] == []


def test_tags_propagated():
    fm = build_frontmatter(_entry(tags=["travel", "food"]))
    assert fm["tags"] == ["travel", "food"]


def test_missing_values_are_none_not_empty_string():
    fm = build_frontmatter(_entry())
    assert fm["device"] is None
    assert fm["duration_seconds"] is None


# TC-FM-10, 11, 12
import pytest

@pytest.mark.parametrize("place,locality,expected_name", [
    ("Boracay", None, "Boracay"),
    (None, "Aklan", "Aklan"),
    ("Boracay", "Malay", "Boracay, Malay"),
])
def test_location_name_combinations(place, locality, expected_name):
    entry = _entry(location=DayOneLocation(placeName=place, localityName=locality))
    fm = build_frontmatter(entry)
    assert fm["location"]["name"] == expected_name


# TC-FM-13
def test_location_partial_coords_is_none():
    entry = _entry(location=DayOneLocation(latitude=14.55, longitude=None))
    fm = build_frontmatter(entry)
    assert fm["location"]["coords"] is None


# TC-FM-14
def test_location_user_label():
    entry = _entry(location=DayOneLocation(userLabel="Home"))
    fm = build_frontmatter(entry)
    assert fm["location"]["label"] == "Home"


# TC-FM-15
def test_duration_seconds_propagated():
    entry = _entry(duration_seconds=120)
    fm = build_frontmatter(entry)
    assert fm["duration_seconds"] == 120


# TC-FM-16
def test_weather_humidity_zero_preserved():
    from jkb.models.dayone import DayOneWeather
    entry = _entry(weather=DayOneWeather(relativeHumidity=0))
    fm = build_frontmatter(entry)
    assert fm["weather"]["humidity"] == 0


# TC-FM-17
def test_starred_and_pinned_propagated():
    entry = _entry(starred=True, pinned=True)
    fm = build_frontmatter(entry)
    assert fm["starred"] is True
    assert fm["pinned"] is True
