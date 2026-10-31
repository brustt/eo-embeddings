# Slice land-cover check (issue #9)

Purpose: catch a bad `tessera-rgb` before the hours-long read. The agent prepared this record; every result cell is owner-only and stays `PENDING (owner)` until the owner looks.

## Reference

- Land-cover reference: **OSO** (Theia / CESBIO / CNES), **OSO 2023** edition (year 2023, 10 m, 24 classes, mainland France).
- Why 2023: the ticket asks for 2025 or the latest before it. Research on 2026-10-30 found no confirmed OSO 2024 or 2025 release. 2022 was released in July 2023, and 2023 was delayed (announced 5 March 2024) then published in the 2016-2023 OSO deposit (12 Sept 2025). OSO >= 2022 exists, so WorldCover is not used.
- Sources:
  - Product page: https://www.theia-land.fr/en/blog/product/land-cover-map/
  - Delay notice for 2023: https://www.theia-land.fr/en/blog/oso-2023-data-a-little-more-patience-ahead/
  - Deposit 2016-2023 (includes OSO_20230101_RASTER): https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi%3A10.57745%2FUZ2NJ7
  - Download: CNES GEODES, https://geodes-portal.cnes.fr/ (search data type "OSO"). If GEODES now offers 2024 or 2025, use the newest and edit the year here.

## Places and results

| # | Place | Expected OSO class | lat | lon | tessera-rgb (`outputs/indre-et-loire-2025/tessera-rgb.tif`) | tessera-pca (`outputs/indre-et-loire-slice/tessera-pca.tif`) |
|---|-------|--------------------|-----|-----|------|------|
| 1 | Forest of Chinon | forest | 47.21 | 0.40 | pass | not checked (outside slice) |
| 2 | Champeigne cereal area | crop | 47.28 | 0.95 | pass | not checked (outside slice) |
| 3 | Central Tours | urban | 47.39 | 0.69 | pass | PENDING (owner) - slice fit, not a result |
| 4 | Loire at Tours | river / water | 47.40 | 0.69 | pass | fail - slice fit, not a result |
| 5 | Vouvray vineyards | vineyard / crop | 47.41 | 0.80 | pass | not checked (outside slice) |

Each result cell takes `pass` or `fail` (plus a short note on the colours seen).

Layer verdicts:

| Layer | Verdict |
|-------|---------|
| tessera-rgb | pass |
| tessera-pca (slice fit, not a result) | fail |
| Check as a whole (passes if at least one Layer passes) | pass|

## Pass rule

A Layer passes when the forest, crop, urban and river places have visibly different colours in it. The check as a whole passes when at least one Layer passes.

## Where to look

- `tessera-rgb`: `outputs/indre-et-loire-2025/tessera-rgb.tif` (EPSG:4326, whole department, all five places inside).
- `tessera-pca`: `outputs/indre-et-loire-slice/tessera-pca.tif` (EPSG:32631, a 5 km box, UTM x 323040-328180, y 5249130-5254290). Only Central Tours (x 325667, y 5251092) and the Loire at Tours (x 325700, y 5252203) fall inside. Chinon, Champeigne and Vouvray (x 334031) are outside, so tessera-pca is not checked there.
- Compare against OSO 2023 at the same coordinates (QGIS: load the OSO raster and the TIFF, jump to lat/lon).
- Helper previews (agent-made, gitignored), about 1 km half-height crops centred on each place, in `outputs/land-cover-check/`:
  - `rgb-forest-chinon.png`, `rgb-champeigne-cereal.png`, `rgb-central-tours.png`, `rgb-loire-tours.png`, `rgb-vouvray-vineyards.png`
  - `pca-central-tours.png`, `pca-loire-tours.png` (about 2 km crop, 3x upscaled; slice fit, not a result)
  These are only a convenience; judge from the TIFFs if in doubt.
