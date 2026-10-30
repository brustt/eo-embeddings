"""Read Tessera embeddings one Block at a time."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version

import numpy as np
from geotessera.registry import zarr_store_url
from geotessera.store import GeoTesseraZarr
from geotessera.zarr import tile_zone
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

CACHE_MAX_SIZE = 2 * 1024**3
STORE_VERSION = "v1.1"


class UtmZoneError(ValueError):
    """The box's west and east edges fall in different UTM zones."""


class MaskSourceError(ValueError):
    """The store's mask source is not `source_nodata`."""


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    height: int
    width: int
    bands: int
    block_size: int
    year: int
    store_url: str
    geotessera_version: str
    mask_source: str

    def blocks(self) -> Iterator[tuple[int, int]]:
        """(row, col) block indices, including a partial last row and column."""
        n_rows = -(-self.height // self.block_size)
        n_cols = -(-self.width // self.block_size)
        for r in range(n_rows):
            for c in range(n_cols):
                yield r, c


def check_mask_source(attrs) -> str:
    value = attrs.get("geotessera:mask_source")
    if value != "source_nodata":
        raise MaskSourceError(
            f"geotessera:mask_source is {value!r}, expected 'source_nodata'"
        )
    return value


def _snap(centres: np.ndarray, lo: float, hi: float, px: float, descending: bool):
    """Index range [i0, i1) of pixels whose extent intersects [lo, hi]."""
    c = -centres if descending else centres
    a, b = (-hi, -lo) if descending else (lo, hi)
    i0 = int(np.searchsorted(c, a - px / 2, "right"))
    i1 = int(np.searchsorted(c, b + px / 2, "left"))
    i0, i1 = max(i0, 0), min(i1, len(c))
    if i1 <= i0:
        raise IndexError("Box does not intersect the zone grid")
    return i0, i1


def open_reader(bbox_wgs84, year: int, cache_dir, block_size: int = 512):
    west, south, east, north = bbox_wgs84
    zone_w, zone_e = tile_zone(west), tile_zone(east)
    if zone_w != zone_e:
        raise UtmZoneError(f"Box spans UTM zones {zone_w} and {zone_e}")
    url = zarr_store_url(STORE_VERSION)
    store = GeoTesseraZarr(url, cache_dir=cache_dir, cache_max_size=CACHE_MAX_SIZE)
    ds = store.open_zone(zone=zone_w)
    mask_source = check_mask_source(ds.attrs)
    crs = f"EPSG:{32600 + zone_w}"
    x0b, y0b, x1b, y1b = transform_bounds(
        "EPSG:4326", crs, west, south, east, north, densify_pts=21
    )
    xs, ys = ds["x"].values, ds["y"].values
    px = float(xs[1] - xs[0])
    c0, c1 = _snap(xs, x0b, x1b, px, False)
    r0, r1 = _snap(ys, y0b, y1b, px, True)
    grid = Grid(
        crs=crs,
        transform=Affine(px, 0, xs[c0] - px / 2, 0, -px, ys[r0] + px / 2),
        height=r1 - r0,
        width=c1 - c0,
        bands=int(ds["embeddings"].sizes["band"]),
        block_size=block_size,
        year=year,
        store_url=url,
        geotessera_version=_pkg_version("geotessera"),
        mask_source=mask_source,
    )
    return TesseraReader(grid, ds)


class TesseraReader:
    def __init__(self, grid: Grid, zone_ds):
        self.grid = grid
        self._ds = zone_ds

    def read_window(self, row: int, col: int):
        g = self.grid
        r0, c0 = row * g.block_size, col * g.block_size
        r1, c1 = min(r0 + g.block_size, g.height), min(c0 + g.block_size, g.width)
        # Selection is by pixel centre, so pass first and last centres.
        x_first, y_first = g.transform @ (c0 + 0.5, r0 + 0.5)
        x_last, y_last = g.transform @ (c1 - 0.5, r1 - 0.5)
        bbox = (x_first, y_last, x_last, y_first)
        emb, scales, tr = self._ds.tessera.read_region_quantized(bbox, g.year)
        h, w = r1 - r0, c1 - c0
        if emb.shape != (h, w, g.bands) or scales.shape != (h, w):
            raise ValueError(
                f"Unexpected shape {emb.shape}/{scales.shape}, expected {(h, w, g.bands)}"
            )
        expected = g.transform @ Affine.translation(c0, r0)
        if not np.allclose(tuple(tr)[:6], tuple(expected)[:6]):
            raise ValueError(f"Unexpected transform {tr}, expected {expected}")
        return emb.transpose(2, 0, 1), scales
