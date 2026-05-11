from __future__ import annotations
import re
from jkb.models.entry import NormalizedEntry

_DAYONE_IMG_RE = re.compile(r'!\[([^\]]*)\]\(dayone-moment://([^)]+)\)')
_INLINE_TAG_RE = re.compile(r'(?<!\w)#([a-zA-Z][a-zA-Z0-9_-]*)')
_IMAGE_REF_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)')


def _sanitize_tag(tag: str) -> str:
    """Lowercase, replace spaces with hyphens, strip non-alphanumeric except hyphens/underscores."""
    tag = tag.lower().replace(" ", "-")
    tag = re.sub(r'[^a-z0-9_-]', '', tag)
    return tag


def _rewrite_images(text: str, attachment_map: dict[str, str]) -> str:
    """Replace dayone-moment:// refs with attachment paths or missing-attachment comments."""
    def replacer(m: re.Match) -> str:
        caption = m.group(1)
        identifier = m.group(2)
        if identifier in attachment_map:
            path = attachment_map[identifier]
            return f"![{caption}]({path})"
        return f"<!-- missing attachment: {identifier} -->"

    return _DAYONE_IMG_RE.sub(replacer, text)


def _extract_and_move_images(text: str) -> tuple[str, str]:
    """
    Extract all image references from text, remove them from their original positions.
    Returns (image_block, remaining_text).
    image_block is a newline-joined string of image refs, or "" if none.
    """
    images = _IMAGE_REF_RE.findall(text)
    if not images:
        return "", text

    # Remove all image refs from text
    remaining = _IMAGE_REF_RE.sub("", text)

    # Collapse excess blank lines left by removals
    remaining = re.sub(r'\n{3,}', '\n\n', remaining)
    remaining = remaining.strip()

    image_block = "\n".join(images)
    return image_block, remaining


def _collect_inline_tags(text: str) -> set[str]:
    """Find Obsidian-style tags already present in the text."""
    return set(_INLINE_TAG_RE.findall(text))


def _append_tags(text: str, tags: list[str]) -> str:
    """Append any tags not already inline at the end of the body."""
    inline = _collect_inline_tags(text)
    missing = [t for t in tags if _sanitize_tag(t) not in {s.lower() for s in inline}]
    if not missing:
        return text
    tag_line = " ".join(f"#{_sanitize_tag(t)}" for t in missing if _sanitize_tag(t))
    if not tag_line.strip():
        return text
    if text:
        return text + "\n\n" + tag_line
    return tag_line


def _normalize_whitespace(text: str) -> str:
    """Strip trailing whitespace per line, collapse 3+ blank lines to 2, ensure one trailing newline."""
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines to 2 (at most one blank line between content)
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    # Ensure exactly one trailing newline
    text = "\n".join(result).rstrip("\n") + "\n"
    return text


def build_body(entry: NormalizedEntry) -> str:
    """
    Returns the final Markdown body string (no frontmatter).
    Assumes entry.attachment_map is already populated by the attachment handler.
    """
    text = entry.text or ""

    # Step 1: Rewrite image references
    text = _rewrite_images(text, entry.attachment_map)

    # Step 2: Extract images and move to top
    image_block, text = _extract_and_move_images(text)

    # Step 3 & 4: Append missing tags
    text = _append_tags(text, entry.tags)

    # Step 5: Assemble with images at top
    if image_block:
        if text:
            body = image_block + "\n\n" + text
        else:
            body = image_block
    else:
        body = text

    # Step 6: Normalize whitespace
    return _normalize_whitespace(body)
