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

## Run the pilot slice

From a clean clone (needs [uv](https://docs.astral.sh/uv/) and network access to the Tessera store):

```
uv sync --group pilot
uv run --group pilot python -m pilot.crop_rgb
uv run --group pilot python -m pilot.build_cube --slice
uv run --group pilot python -m pilot.make_pca --slice
uv run --group pilot pilot/check_outputs.py --slice
```

The boundary is committed (`pilot/data/indre-et-loire.geojson`), so `pilot.fetch_boundary` is only
needed to refresh it. Outputs land in `outputs/` and `data/` (both gitignored). The full run uses the
same steps without `--slice`; its manifest is `publish/indre-et-loire-2025.publish.yaml`. The slice
manifest, `pilot/slice.publish.yaml`, stays outside `publish/` and must never be published to production.
