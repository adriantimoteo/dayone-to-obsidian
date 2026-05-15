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


# TC-BODY-10
def test_multiple_images_extracted_to_top_in_order():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B10", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="Para 1\n\n![a](dayone-moment://P1)\n\nPara 2\n\n![b](dayone-moment://P2)",
        attachment_map={"P1": "./attachments/p1.jpeg", "P2": "./attachments/p2.jpeg"},
    )
    body = build_body(entry)
    lines = body.strip().split("\n")
    # First non-empty lines should be the two images in order
    img_lines = [l for l in lines if l.startswith("![")]
    assert img_lines[0].startswith("![a]")
    assert img_lines[1].startswith("![b]")
    # Images should precede body text
    first_img_idx = next(i for i, l in enumerate(lines) if l.startswith("!["))
    para1_idx = next(i for i, l in enumerate(lines) if "Para" in l)
    assert first_img_idx < para1_idx


# TC-BODY-11
def test_tag_sanitizes_to_empty_omitted():
    from jkb.stages.body import _sanitize_tag
    assert _sanitize_tag("!!!") == ""


# TC-BODY-12
def test_tag_dedup_case_insensitive():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B12", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="Today #Travel was fun",
        tags=["travel"],
    )
    body = build_body(entry)
    # #travel should appear exactly once (inline #Travel counts as match)
    assert body.lower().count("#travel") == 1


# TC-BODY-13
def test_tag_with_spaces_sanitized_and_appended_once():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B13", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="No inline tags here",
        tags=["New York"],
    )
    body = build_body(entry)
    assert body.count("#new-york") == 1


# TC-BODY-14
def test_image_from_mid_paragraph_appears_at_top():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B14", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="Look at this ![img](dayone-moment://P1) photo today",
        attachment_map={"P1": "./attachments/img.jpeg"},
    )
    body = build_body(entry)
    lines = [l for l in body.split("\n") if l.strip()]
    # First non-empty line should be the image
    assert lines[0].startswith("![img]")


# TC-BODY-15
def test_no_images_no_tags_body_is_text():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B15", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="Hello world",
    )
    body = build_body(entry)
    assert body.strip() == "Hello world"
    assert body.endswith("\n")


# TC-BODY-16
def test_trailing_whitespace_stripped_from_lines():
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.body import build_body
    from datetime import datetime, timezone
    entry = NormalizedEntry(
        uuid="B16", journal="Test",
        creation_date=datetime(2020,1,1,tzinfo=timezone.utc),
        text="Line 1   \nLine 2\t",
    )
    body = build_body(entry)
    for line in body.split("\n"):
        assert not line.endswith((" ", "\t")), f"trailing whitespace in: {repr(line)}"


# TC-BODY-17
def test_dayone_moment_plain_text_not_matched():
    from jkb.stages.body import _DAYONE_IMG_RE
    text = "see dayone-moment://PHOTO_ID"
    assert _DAYONE_IMG_RE.search(text) is None
