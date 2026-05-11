from __future__ import annotations
import tempfile
import logging
from pathlib import Path

import typer

from jkb.pipeline import process_entry
from jkb.stages.parse import parse
from jkb.stages.validate import validate
from jkb.utils.checkpoint import Checkpoint
from jkb.utils.migration_log import MigrationLog

app = typer.Typer(help="JKB — Personal Journal Knowledge Base tools.")
logger = logging.getLogger(__name__)


@app.callback()
def _callback() -> None:
    """JKB — Personal Journal Knowledge Base tools."""


@app.command()
def migrate(
    input: Path = typer.Argument(..., help="Path to a .dayone ZIP file or directory of .dayone ZIPs"),
    output: Path = typer.Argument(..., help="Directory where output files will be written"),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Resume from checkpoint"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .md files"),
    log_path: Path = typer.Option(None, "--log", help="Path for migration-log.md"),
    checkpoint_path: Path = typer.Option(None, "--checkpoint", help="Path for checkpoint file"),
) -> None:
    """Migrate DayOne journal export(s) to structured Markdown."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    output.mkdir(parents=True, exist_ok=True)
    log_out = log_path or output / "migration-log.md"
    chk_path = checkpoint_path or output / ".migration-checkpoint.json"

    checkpoint = Checkpoint(chk_path)
    if not resume:
        checkpoint.clear()

    migration_log = MigrationLog()
    processed = 0

    with tempfile.TemporaryDirectory() as staging_str:
        staging = Path(staging_str)

        entries = parse(input, staging)
        validated = validate(entries, staging)

        try:
            for entry, result in validated:
                try:
                    process_entry(
                        entry, result, staging, output,
                        checkpoint, migration_log, overwrite,
                    )
                except Exception as e:
                    logger.error("Unhandled error on entry %s: %s", entry.uuid, e)

                processed += 1
                if processed % 100 == 0:
                    typer.echo(f"[jkb] {processed} entries processed...", err=True)

        except KeyboardInterrupt:
            typer.echo("\n[INTERRUPTED] Checkpoint saved. Re-run with --resume to continue.", err=True)
        finally:
            migration_log.write(log_out)
            typer.echo(f"Migration log written to {log_out}")

    typer.echo(f"Done. {processed} entries processed.")
