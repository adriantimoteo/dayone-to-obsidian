# JKB — Personal Journal Knowledge Base

A RAG pipeline that migrates DayOne journal entries into a locally-queryable knowledge base. Ask natural-language questions over a decade of personal data and get answers grounded in citations from your own journal.

```
DayOne JSON Export → jkb migrate → Structured Markdown Vault
                                              ↓
                                   jkb index (ChromaDB)
                                              ↓
                                   jkb ask "your question"
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (for local LLM synthesis)

## Setup

```bash
uv sync
ollama pull llama3.2
```

## Usage

### 1. Migrate

Convert a DayOne `.zip` export to structured Markdown:

```bash
jkb migrate /path/to/export.zip /path/to/vault
```

Use `--resume` to continue an interrupted migration. Outputs a `migration-log.md` in the vault.

### 2. Index

Embed the vault into ChromaDB:

```bash
jkb index /path/to/vault
```

### 3. Ask

Query your journal in natural language:

```bash
jkb ask "when was I last in Sagada?"
```

By default this runs fully locally via Ollama. For cloud synthesis:

```bash
# Anthropic (requires ANTHROPIC_API_KEY)
jkb ask "what was I working on in 2021?" --backend anthropic

# OpenAI-compatible endpoint
jkb ask "summarise my goals from last year" --backend openai
```

Use `--verbose` to see the retrieved chunks before the answer. Use `--k` to control how many chunks are retrieved (default: 10).

Set `JKB_VAULT` to your vault path to avoid passing `--vault` every time.

## Development

```bash
uv run pytest                  # full test suite
uv run pytest -x --tb=short    # stop on first failure
uv run ruff check src tests    # lint
```
