from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DayOneLocation(BaseModel):
    model_config = {"extra": "allow"}

    latitude: float | None = None
    longitude: float | None = None
    placeName: str | None = None
    localityName: str | None = None
    country: str | None = None
    administrativeArea: str | None = None
    userLabel: str | None = None


class DayOneWeather(BaseModel):
    model_config = {"extra": "allow"}

    temperatureCelsius: float | None = None
    conditionDescription: str | None = None
    windSpeedKPH: float | None = None
    relativeHumidity: int | None = None
    weatherCode: str | None = None
    windBearing: int | None = None
    sunriseDate: str | None = None
    sunsetDate: str | None = None


class DayOneActivity(BaseModel):
    model_config = {"extra": "allow"}

    stepCount: int | None = None
    activityName: str | None = None
    heartRate: int | None = None
    distance: float | None = None


class DayOnePhoto(BaseModel):
    model_config = {"extra": "allow"}

    identifier: str
    md5: str | None = None
    type: str = "jpeg"
    width: int | None = None
    height: int | None = None
    date: str | None = None
    caption: str | None = None
    isSketch: bool | None = None


class DayOneVideo(BaseModel):
    model_config = {"extra": "allow"}

    identifier: str
    md5: str | None = None
    type: str = "mp4"
    width: int | None = None
    height: int | None = None
    duration: int | None = None


class DayOneAudio(BaseModel):
    model_config = {"extra": "allow"}

    identifier: str
    md5: str | None = None
    type: str = "m4a"
    duration: int | None = None


class DayOnePDF(BaseModel):
    model_config = {"extra": "allow"}

    identifier: str
    md5: str | None = None
    type: str = "pdf"
    pdfName: str | None = None


class DayOneEntry(BaseModel):
    model_config = {"extra": "allow"}

    uuid: str
    text: str | None = None
    richText: str | None = None
    creationDate: datetime
    modifiedDate: datetime | None = None
    timeZone: str = "UTC"
    starred: bool = False
    pinned: bool = False
    tags: list[str] = Field(default_factory=list)
    location: DayOneLocation | None = None
    weather: DayOneWeather | None = None
    activity: DayOneActivity | None = None
    photos: list[DayOnePhoto] = Field(default_factory=list)
    videos: list[DayOneVideo] = Field(default_factory=list)
    audios: list[DayOneAudio] = Field(default_factory=list)
    pdfAttachments: list[DayOnePDF] = Field(default_factory=list)
    deviceName: str | None = None
    creationOSName: str | None = None
    creationOSVersion: str | None = None
    duration: int | None = None
    sourceString: str | None = None


class DayOneJournal(BaseModel):
    """Wrapper for a DayOne JSON export file."""

    model_config = {"extra": "allow"}

    entries: list[DayOneEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
