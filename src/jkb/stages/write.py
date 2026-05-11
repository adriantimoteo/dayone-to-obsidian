from __future__ import annotations
import os
import re
import tempfile
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from jkb.models.entry import NormalizedEntry

_ISO_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def _quote_datetime_strings(data: dict) -> dict:
    """Wrap ISO 8601 datetime strings in DoubleQuotedScalarString to prevent YAML timestamp coercion."""
    result = {}
    for k, v in data.items():
        if isinstance(v, str) and _ISO_RE.match(v):
            result[k] = DoubleQuotedScalarString(v)
        elif isinstance(v, dict):
            result[k] = _quote_datetime_strings(v)
        else:
            result[k] = v
    return result


def _render_yaml(data: dict) -> str:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)
    buf = StringIO()
    yaml.dump(data, buf)
    return buf.getvalue()


def _local_datetime(entry: NormalizedEntry):
    """Return creation_date in entry's local timezone, falling back to UTC."""
    try:
        tz = ZoneInfo(entry.timezone)
        return entry.creation_date.astimezone(tz)
    except (ZoneInfoNotFoundError, Exception):
        return entry.creation_date


def _filename(entry: NormalizedEntry) -> str:
    local_dt = _local_datetime(entry)
    date_part = local_dt.strftime("%Y-%m-%d-%H%M")
    hash_part = entry.uuid[:8].lower()
    return f"{date_part}-{hash_part}.md"


def write_entry(
    entry: NormalizedEntry,
    frontmatter: dict,
    body: str,
    output_root: Path,
    overwrite: bool = False,
) -> Path | None:
    """
    Assembles and writes a .md file.
    Returns the written file path on success, None if skipped (file exists and overwrite=False).
    Caller is responsible for checking ValidationResult.is_valid before calling.
    """
    local_dt = _local_datetime(entry)
    year = local_dt.strftime("%Y")
    month = local_dt.strftime("%m")

    output_dir = output_root / year / month
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = _filename(entry)
    file_path = output_dir / filename

    if file_path.exists() and not overwrite:
        return None

    frontmatter = _quote_datetime_strings(frontmatter)
    yaml_str = _render_yaml(frontmatter)
    content = f"---\n{yaml_str}---\n\n{body}"

    # Atomic write: write to temp, then rename
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path_str, file_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    return file_path
