from __future__ import annotations
import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jkb.index.embedder import get_embedder
from jkb.index.manifest import Manifest
from jkb.index.pipeline import run_index
from jkb.index.vectorstore import VectorStore
from jkb.pipeline import process_entry
from jkb.query.search import HybridSearcher
from jkb.query.synthesizer import Synthesizer
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


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question to ask your journal"),
    model: str = typer.Option("claude-haiku-4-5-20251001", "--model", help="LLM model to use for synthesis"),
    backend: str = typer.Option("anthropic", "--backend", help="LLM backend: anthropic | openai"),
    base_url: str = typer.Option(None, "--base-url", help="Base URL for OpenAI-compatible endpoints"),
    k: int = typer.Option(10, "--k", help="Number of chunks to retrieve"),
    vault: Path = typer.Option(None, "--vault", help="Path to the vault directory (default: $JKB_VAULT or ./.chroma)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieved chunks before the answer"),
) -> None:
    """Ask a natural-language question and get an answer grounded in your journal."""
    console = Console()

    vault_path: Path
    if vault is not None:
        vault_path = vault
    else:
        env_vault = os.environ.get("JKB_VAULT")
        vault_path = Path(env_vault) if env_vault else Path(".chroma")

    chroma_path = vault_path if vault_path.name == ".chroma" else vault_path / ".chroma"

    if not chroma_path.exists():
        typer.echo(
            f"Error: chroma directory does not exist: {chroma_path}\n"
            "Run `jkb index <vault>` first to build the index.",
            err=True,
        )
        raise typer.Exit(code=1)

    from jkb.query.backends import AnthropicBackend, OpenAIBackend  # noqa: PLC0415

    try:
        if backend == "anthropic":
            llm_backend = AnthropicBackend()
        elif backend == "openai":
            llm_backend = OpenAIBackend(base_url=base_url)
        else:
            typer.echo(f"Error: unknown backend {backend!r}. Choose from: anthropic, openai", err=True)
            raise typer.Exit(code=1)
    except (ImportError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    console.print("Searching your journal...", style="bold")

    store = VectorStore(chroma_path)
    embedder = get_embedder("nomic")
    searcher = HybridSearcher(store, embedder)

    results = searcher.search(question, k=k)

    if not results:
        console.print("[yellow]No results found in your journal for that query.[/yellow]")
        raise typer.Exit(code=0)

    if verbose:
        table = Table(title="Retrieved Chunks", show_lines=True)
        table.add_column("Entry ID", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta", justify="right")
        table.add_column("Excerpt")
        for r in results:
            excerpt = r.text[:200].replace("\n", " ")
            if len(r.text) > 200:
                excerpt += "..."
            table.add_row(r.entry_id, f"{r.score:.3f}", excerpt)
        console.print(table)

    synthesizer = Synthesizer(llm_backend, model=model)
    synthesis = synthesizer.synthesize(question, results)

    console.print(Panel(synthesis.answer, title="Answer", border_style="green"))

    if synthesis.sources:
        console.print("\n[bold]Sources:[/bold]")
        for source in synthesis.sources:
            console.print(f"  • {source}")
