from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from jkb.models.dayone import (
    DayOneLocation,
    DayOneWeather,
    DayOneActivity,
    DayOnePhoto,
    DayOneVideo,
    DayOneAudio,
    DayOnePDF,
)


def build_device_string(
    device_name: str | None,
    os_name: str | None,
    os_version: str | None,
) -> str | None:
    """Returns 'iPhone 13 Pro / iOS 16.1' or None if all parts absent."""
    parts = []
    if device_name:
        parts.append(device_name)
    if os_name and os_version:
        parts.append(f"{os_name} {os_version}")
    elif os_name:
        parts.append(os_name)
    return " / ".join(parts) if parts else None


class NormalizedEntry(BaseModel):
    uuid: str
    journal: str
    text: str = ""
    creation_date: datetime
    modified_date: datetime | None = None
    timezone: str = "UTC"
    starred: bool = False
    pinned: bool = False
    tags: list[str] = Field(default_factory=list)
    device: str | None = None
    location: DayOneLocation | None = None
    weather: DayOneWeather | None = None
    activity: DayOneActivity | None = None
    photos: list[DayOnePhoto] = Field(default_factory=list)
    videos: list[DayOneVideo] = Field(default_factory=list)
    audios: list[DayOneAudio] = Field(default_factory=list)
    pdfs: list[DayOnePDF] = Field(default_factory=list)
    duration_seconds: int | None = None
    source_string: str | None = None
    attachment_map: dict[str, str] = Field(default_factory=dict)
