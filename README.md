# eo-embeddings

A library for extraction and storage of Earth-observation embeddings: it reads
Tessera embeddings and stores them as local cubes. It does not publish or
analyse them.

Pinned `geotessera` version: **0.11.0** (exact; the code relies on its method
names and each cube records its version).

## Development

```
uv sync --all-groups
uv run ruff check
uv run pytest
```
