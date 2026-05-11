"""End-to-end integration tests for the full migration pipeline."""
import json
import zipfile
from pathlib import Path
from typer.testing import CliRunner

from jkb.cli import app


def _make_dayone_zip(zip_path: Path, journal_name: str, entries: list[dict],
                     photos: dict[str, bytes] | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{journal_name}.json", json.dumps({"entries": entries}))
        if photos:
            for fname, data in photos.items():
                zf.writestr(f"photos/{fname}", data)


def _entry(uuid: str, date: str = "2020-06-15T10:30:00Z", text: str = "Hello journal") -> dict:
    return {
        "uuid": uuid,
        "creationDate": date,
        "text": text,
        "timeZone": "UTC",
        "starred": False,
        "pinned": False,
        "tags": ["test"],
    }


runner = CliRunner()


def test_e2e_basic_migration(tmp_path):
    zip_path = tmp_path / "input" / "Test.dayone"
    _make_dayone_zip(zip_path, "Test", [_entry("UUID0001"), _entry("UUID0002")])
    output = tmp_path / "output"

    result = runner.invoke(app, ["migrate", str(zip_path), str(output)])
    assert result.exit_code == 0, result.output

    md_files = [f for f in output.rglob("*.md") if f.name != "migration-log.md"]
    assert len(md_files) == 2
    assert (output / "migration-log.md").exists()


def test_e2e_log_written(tmp_path):
    zip_path = tmp_path / "input" / "Test.dayone"
    _make_dayone_zip(zip_path, "Test", [_entry("UUID0010")])
    output = tmp_path / "output"

    runner.invoke(app, ["migrate", str(zip_path), str(output)])

    log_content = (output / "migration-log.md").read_text(encoding="utf-8")
    assert "# Migration Log" in log_content
    assert "| Total attempted | 1 |" in log_content


def test_e2e_resume(tmp_path):
    zip_path = tmp_path / "input" / "Test.dayone"
    _make_dayone_zip(zip_path, "Test", [_entry("R001"), _entry("R002"), _entry("R003")])
    output = tmp_path / "output"
    chk = tmp_path / "cp.json"

    # First run — process all
    runner.invoke(app, ["migrate", str(zip_path), str(output), "--checkpoint", str(chk)])
    count_first = len([f for f in output.rglob("*.md") if f.name != "migration-log.md"])

    # Second run with --resume — nothing new should be written
    runner.invoke(app, ["migrate", str(zip_path), str(output), "--resume", "--checkpoint", str(chk)])
    count_second = len([f for f in output.rglob("*.md") if f.name != "migration-log.md"])

    assert count_first == 3
    assert count_second == 3  # no duplicates


def test_e2e_duplicate_uuid_skipped(tmp_path):
    entries = [_entry("DUP001"), _entry("DUP001")]  # same UUID twice
    zip_path = tmp_path / "input" / "Test.dayone"
    _make_dayone_zip(zip_path, "Test", entries)
    output = tmp_path / "output"

    runner.invoke(app, ["migrate", str(zip_path), str(output)])

    md_files = [f for f in output.rglob("*.md") if f.name != "migration-log.md"]
    assert len(md_files) == 1  # only one written


def test_e2e_missing_photo_no_crash(tmp_path):
    entry = _entry("P001")
    entry["photos"] = [{"identifier": "PHOTO1", "md5": "abc123", "type": "jpeg"}]
    zip_path = tmp_path / "input" / "Test.dayone"
    _make_dayone_zip(zip_path, "Test", [entry])  # no actual photo in ZIP
    output = tmp_path / "output"

    result = runner.invoke(app, ["migrate", str(zip_path), str(output)])
    assert result.exit_code == 0
    md_files = [f for f in output.rglob("*.md") if f.name != "migration-log.md"]
    assert len(md_files) == 1  # still written despite missing photo
