from datetime import datetime, timezone
from jkb.models.entry import NormalizedEntry
from jkb.stages.body import build_body, _sanitize_tag


def _entry(text: str = "", tags: list[str] | None = None, attachment_map: dict | None = None, **kwargs) -> NormalizedEntry:
    return NormalizedEntry(
        uuid="abc123",
        journal="Test",
        creation_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text=text,
        tags=tags or [],
        attachment_map=attachment_map or {},
        **kwargs,
    )


def test_empty_entry_no_text_no_tags():
    result = build_body(_entry())
    assert result.strip() == ""


def test_plain_text_preserved():
    result = build_body(_entry(text="Hello world"))
    assert "Hello world" in result


def test_dayone_image_rewritten():
    attachment_map = {"PHOTO_ID": "./attachments/abc123.jpeg"}
    entry = _entry(
        text="Look at this: ![my photo](dayone-moment://PHOTO_ID)",
        attachment_map=attachment_map,
    )
    result = build_body(entry)
    assert "./attachments/abc123.jpeg" in result
    assert "dayone-moment://" not in result


def test_missing_dayone_image_becomes_comment():
    entry = _entry(text="![alt](dayone-moment://MISSING_ID)")
    result = build_body(entry)
    assert "<!-- missing attachment: MISSING_ID -->" in result


def test_images_moved_to_top():
    attachment_map = {"P1": "./attachments/img1.jpeg"}
    entry = _entry(
        text="Some text\n\n![img](dayone-moment://P1)\n\nMore text",
        attachment_map=attachment_map,
    )
    result = build_body(entry)
    img_pos = result.index("![img]")
    text_pos = result.index("Some text")
    assert img_pos < text_pos


def test_tags_appended():
    entry = _entry(text="Hello world", tags=["travel", "food"])
    result = build_body(entry)
    assert "#travel" in result
    assert "#food" in result


def test_existing_inline_tags_not_duplicated():
    entry = _entry(text="Going #travel today", tags=["travel", "food"])
    result = build_body(entry)
    # #travel already inline, only #food should be appended
    assert result.count("#travel") == 1
    assert "#food" in result


def test_tag_sanitization():
    assert _sanitize_tag("My Tag") == "my-tag"
    assert _sanitize_tag("food & drink") == "food--drink"
    assert _sanitize_tag("Travel") == "travel"


def test_trailing_newline():
    result = build_body(_entry(text="Hello"))
    assert result.endswith("\n")


def test_no_triple_blank_lines():
    entry = _entry(text="Line 1\n\n\n\n\nLine 2")
    result = build_body(entry)
    assert "\n\n\n" not in result


def test_media_only_entry():
    attachment_map = {"P1": "./attachments/img.jpeg"}
    entry = _entry(
        text="![](dayone-moment://P1)",
        tags=["photo"],
        attachment_map=attachment_map,
    )
    result = build_body(entry)
    assert "![](./attachments/img.jpeg)" in result
    assert "#photo" in result
