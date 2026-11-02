"""Shared helpers for the Tessera pilot."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from geotessera.zarr import valid_scale_mask
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


def to_cog(tmp, path, resampling=OVERVIEW_RESAMPLING) -> None:
    """Copy a tiled GTiff to a COG and delete the GTiff."""
    rio_copy(tmp, path, driver="COG", overview_resampling=resampling, **COG_OPTIONS)
    Path(tmp).unlink()


def write_cog(
    path,
    data: np.ndarray,
    transform: Affine,
    nodata=0,
    crs="EPSG:4326",
    resampling=OVERVIEW_RESAMPLING,
) -> None:
    """Write (bands, h, w) uint8 data as a COG (EPSG:4326 and nearest overviews by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    profile = dict(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    to_cog(tmp, path, resampling)


PCA_SAMPLE = 10_000_000
SEED = 0
SLICE_OUTPUT_DIR = ROOT / "outputs" / "indre-et-loire-slice"
CACHE_DIR = ROOT / "data" / "cache"
CUBE_DIR = ROOT / "data" / "cube"


def apply_sign_rule(components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip each component so its largest loading (by magnitude) is positive."""
    top = components[np.arange(len(components)), np.abs(components).argmax(axis=1)]
    signs = np.where(top < 0, -1, 1)
    return components * signs[:, None], signs


def stretch(values: np.ndarray, p2, p98) -> np.ndarray:
    """Map p2..p98 to 1..255 (rounded, clipped) as uint8."""
    if p98 == p2:
        raise ValueError(f"p98 equals p2 ({p2}); cannot stretch")
    scaled = np.rint(1 + 254 * (values - p2) / (p98 - p2))
    return np.clip(scaled, 1, 255).astype(np.uint8)


def sample_pixels(
    blocks, total: int = PCA_SAMPLE, seed: int = SEED, n_blocks: int | None = None
) -> np.ndarray:
    """Sample (n, bands) pixels from per-Block arrays, in the order given.

    `blocks` may be a lazy iterator if `n_blocks` is given. Each Block gives `ceil(total / n_blocks)` pixels, or all of them if fewer, drawn
    without replacement. The pool is shuffled and cut to `total`.
    """
    if n_blocks is None:
        blocks = list(blocks)
        n_blocks = len(blocks)
    rng = np.random.default_rng(seed)
    quota = -(-total // n_blocks)
    pool, filled = None, 0  # one preallocated array: no concatenate or permuted copy
    for px in blocks:
        if pool is None:
            pool = np.empty((quota * n_blocks, px.shape[1]), dtype=px.dtype)
        n = min(quota, len(px))
        pool[filled:filled + n] = px[rng.choice(len(px), size=n, replace=False)] if n else px[:0]
        filled += n
    pool = pool[:filled]
    rng.shuffle(pool)  # in place
    return pool[:total]


def fit_pca(sample: np.ndarray, n_components: int = 3, chunk: int = 1_000_000):
    """Top principal components from the covariance of `sample`, in float64 chunks.

    Returns (mean, components, explained_variance, explained_variance_ratio). The same
    result as a full SVD up to component sign, but memory is one chunk, not several
    copies of the sample.
    """
    n, bands = sample.shape
    mean = sample.mean(axis=0, dtype=np.float64)
    cov = np.zeros((bands, bands))
    for i in range(0, n, chunk):
        x = sample[i:i + chunk].astype(np.float64) - mean
        cov += x.T @ x
    cov /= n - 1
    values, vectors = np.linalg.eigh(cov)  # ascending
    top = np.argsort(values)[::-1][:n_components]
    return mean, vectors[:, top].T, values[top], values[top] / np.trace(cov)


def project(sample: np.ndarray, mean, components, chunk: int = 1_000_000) -> np.ndarray:
    """(n, components) float64 projection of `sample`, computed in chunks."""
    return np.concatenate([
        (sample[i:i + chunk].astype(np.float64) - mean) @ components.T
        for i in range(0, len(sample), chunk)
    ])


def pca_pixels(emb, scales, inside, mean, components, p2, p98) -> np.ndarray:
    """(bands, h, w) float embeddings to (3, h, w) uint8; 0 where outside or no data."""
    valid = inside & valid_scale_mask(scales) & np.isfinite(emb).all(axis=0)
    out = np.zeros((len(components), *inside.shape), dtype=np.uint8)
    if valid.any():
        proj = (emb[:, valid].T - mean) @ components.T
        for i in range(len(components)):
            out[i][valid] = stretch(proj[:, i], p2[i], p98[i])
    return out
