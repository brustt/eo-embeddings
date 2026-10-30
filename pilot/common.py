"""Shared helpers for the Tessera pilot."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine
from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = ROOT / "pilot" / "data" / "indre-et-loire.geojson"
YEAR = 2025
OUTPUT_DIR = ROOT / "outputs" / "indre-et-loire-2025"
RGB_COG = OUTPUT_DIR / "tessera-rgb.tif"
# west, south, east, north: 5 km centred on Tours, crossing the Loire.
SLICE_BOX = (0.656, 47.373, 0.722, 47.418)

COG_OPTIONS = {
    "BLOCKSIZE": 256,
    "COMPRESS": "DEFLATE",
    "LEVEL": 6,
    "BIGTIFF": "IF_SAFER",
    "PREDICTOR": "YES",
}
OVERVIEW_RESAMPLING = "NEAREST"  # every value at every zoom is a source value


def pixel_window(box, px: float, origin) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1), end-exclusive, covering the box.

    Floor of the west and north edges, ceil of the east and south edges.
    `origin` is the (x, y) of the grid's top-left corner.
    """
    west, south, east, north = box
    ox, oy = origin
    return (
        math.floor((oy - north) / px),
        math.ceil((oy - south) / px),
        math.floor((west - ox) / px),
        math.ceil((east - ox) / px),
    )


def check_layout(dimension_names, layout0, shape) -> Affine:
    """Stop unless the store matches what the crop assumes; return the transform."""
    transform = layout0["spatial:transform"]
    if not np.isclose(transform[0], 0.0001) or not np.isclose(transform[4], -0.0001):
        raise ValueError(f"Pixel size is not 0.0001 degrees: {transform}")
    if list(dimension_names) != ["lat", "lon", "band"]:
        raise ValueError(f"Unexpected dimensions {dimension_names}")
    if len(shape) != 3 or shape[2] != 4:
        raise ValueError(f"Band axis must have length 4 (alpha last), got {shape}")
    return Affine(*transform[:6])


def rasterize_boundary(geometry, transform, shape, crs) -> np.ndarray:
    """Boolean mask of pixels whose centre is inside a WGS84 geometry."""
    if str(crs) != "EPSG:4326":
        geometry = transform_geom("EPSG:4326", crs, geometry)
    return rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)


def crop_rgb(rgba: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """(h, w, 4) RGBA and an inside mask to (3, h, w) RGB.

    Pixels outside, or with alpha 0, become (0, 0, 0); the rest keep their values.
    """
    keep = inside & (rgba[..., 3] > 0)
    out = np.where(keep[..., None], rgba[..., :3], 0).astype(np.uint8)
    return out.transpose(2, 0, 1)


def count_zero_pixels(rgb: np.ndarray, counted: np.ndarray) -> tuple[int, int]:
    """Among `counted` pixels of a (3, h, w) array: (all bands 0, some but not all 0)."""
    zeros = (rgb == 0).sum(axis=0)
    return (
        int((counted & (zeros == 3)).sum()),
        int((counted & (zeros > 0) & (zeros < 3)).sum()),
    )


def write_cog(path, data: np.ndarray, transform: Affine, nodata=0) -> None:
    """Write (bands, h, w) uint8 data as an EPSG:4326 COG with nearest overviews."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    profile = dict(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    rio_copy(
        tmp,
        path,
        driver="COG",
        overview_resampling=OVERVIEW_RESAMPLING,
        **COG_OPTIONS,
    )
    tmp.unlink()
