from __future__ import annotations
from jkb.models.entry import NormalizedEntry


def build_frontmatter(entry: NormalizedEntry) -> dict:
    """
    Returns a dict ready for ruamel.yaml serialization.
    All spec-defined keys are always present; missing data is None (→ YAML null).
    """
    # Location
    loc = entry.location
    if loc is not None:
        lat = loc.latitude
        lon = loc.longitude
        coords = [lat, lon] if (lat is not None and lon is not None) else None

        name_parts = [p for p in [loc.placeName, loc.localityName] if p]
        place_name = ", ".join(name_parts) if name_parts else None

        location_dict = {
            "coords": coords,
            "name": place_name,
            "country": loc.country,
            "label": loc.userLabel,
        }
    else:
        location_dict = {
            "coords": None,
            "name": None,
            "country": None,
            "label": None,
        }

    # Weather
    wx = entry.weather
    if wx is not None:
        weather_dict = {
            "temp_c": wx.temperatureCelsius,
            "condition": wx.conditionDescription,
            "wind_kph": wx.windSpeedKPH,
            "humidity": wx.relativeHumidity,
        }
    else:
        weather_dict = {
            "temp_c": None,
            "condition": None,
            "wind_kph": None,
            "humidity": None,
        }

    # Activity
    act = entry.activity
    if act is not None:
        activity_dict = {
            "steps": act.stepCount,
            "motion": act.activityName,
        }
    else:
        activity_dict = {
            "steps": None,
            "motion": None,
        }

    # Date formatting — ISO 8601 with Z suffix (not .isoformat())
    date_str = entry.creation_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    modified_str = entry.modified_date.strftime("%Y-%m-%dT%H:%M:%SZ") if entry.modified_date else None

    return {
        "date": date_str,
        "journal": entry.journal,
        "starred": entry.starred,
        "location": location_dict,
        "weather": weather_dict,
        "activity": activity_dict,
        "device": entry.device,
        "timezone": entry.timezone,
        "tags": entry.tags,
        "people": [],
        "pinned": entry.pinned,
        "duration_seconds": entry.duration_seconds,
        "modified": modified_str,
    }
