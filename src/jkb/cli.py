from __future__ import annotations
import logging
from pathlib import Path

import typer

from jkb.index.embedder import get_embedder
from jkb.index.manifest import Manifest
from jkb.index.pipeline import run_index
from jkb.index.vectorstore import VectorStore
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
    if not input.exists():
        typer.echo(f"Error: input path does not exist: {input}", err=True)
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    output.mkdir(parents=True, exist_ok=True)
    log_out = log_path or output / "migration-log.md"
    chk_path = checkpoint_path or output / ".jkb-checkpoint.json"

    checkpoint = Checkpoint(chk_path)

    migration_log = MigrationLog()
    processed = 0

    staging = output / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

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

    typer.echo(
        f"Migration complete: {migration_log._total_written} written, "
        f"{migration_log._total_skipped} skipped, "
        f"{migration_log._total_warned} with warnings."
    )


@app.command()
def index(
    vault_path: Path = typer.Argument(..., help="Path to the vault directory to index"),
    force_reindex: bool = typer.Option(False, "--force-reindex/--no-force-reindex", help="Re-index all files regardless of hash"),
    model: str = typer.Option("nomic", "--model", help="Embedding model name"),
    chroma_path: Path = typer.Option(None, "--chroma-path", help="Path for Chroma DB (default: <vault>/.chroma)"),
    manifest_path: Path = typer.Option(None, "--manifest-path", help="Path for index manifest (default: <vault>/index-manifest.json)"),
) -> None:
    """Index vault Markdown files into the vector store."""
    if not vault_path.exists():
        typer.echo(f"Error: vault path does not exist: {vault_path}", err=True)
        raise typer.Exit(code=1)

    chroma = chroma_path or vault_path / ".chroma"
    manifest = manifest_path or vault_path / "index-manifest.json"

    typer.echo("Indexing vault...")
    store = VectorStore(chroma)
    mfst = Manifest(manifest)
    embedder = get_embedder(model)
    stats = run_index(vault_path, store, mfst, embedder, force=force_reindex)
    typer.echo(
        f"Index complete: {stats.added} added, {stats.updated} updated, "
        f"{stats.removed} removed, {stats.skipped} skipped."
    )
