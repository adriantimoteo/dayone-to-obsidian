import typer

app = typer.Typer(help="JKB — Personal Journal Knowledge Base tools.")


@app.command()
def migrate(
    input: str = typer.Argument(..., help="Path to .dayone ZIP or directory of ZIPs"),
    output: str = typer.Argument(..., help="Output directory"),
) -> None:
    """Migrate DayOne journal export(s) to structured Markdown."""
    typer.echo("migrate command not yet implemented")
