# Pilot

Scripts for the Indre-et-Loire 2025 pilot. Run from the repo root with `uv run --group pilot`.

- `python -m pilot.fetch_boundary`: writes `data/indre-et-loire.geojson` (committed).
- `python -m pilot.crop_rgb`: writes `outputs/indre-et-loire-2025/tessera-rgb.tif`.
- `python -m pilot.build_cube [--slice]`: writes `data/cube/slice.zarr` (5 km test) or `data/cube/2025.zarr`; prints size and speed.
- `python -m pilot.make_pca [--slice]`: writes `tessera-pca.tif` and `pca_model.json` (slice output goes to `outputs/indre-et-loire-slice/`).

## Boundary

- File: `data/indre-et-loire.geojson`
- Fetched: 2026-10-30
- SHA-256: `916f3ef49e78e726a39ea82a8f62d3482e7388b351e7ce362177e8dff530df83`
- Source: `https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements/37-indre-et-loire/departement-37-indre-et-loire.geojson`
  (a simplified polygon derived from IGN Admin Express).
- Licence: Licence Ouverte / Open Licence 2.0 (Etalab), attribution to IGN.

The planned source, `https://geo.api.gouv.fr/departements/37?format=geojson&geometry=contour`,
answered on 2026-10-30 with only `{"nom", "code"}` and no geometry (HTTP 200), so
`fetch_boundary.py` falls back to the file above. If the API returns a geometry later, re-running
the script uses it (licence Licence Ouverte 2.0, Etalab) and this record must be updated.
