"""Check the manifest, the COGs and the cube.

Run: uv run --group pilot pilot/check_outputs.py [--slice]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from geotessera.zarr import valid_scale_mask
from rasterio.features import bounds as geometry_bounds

from eo_embeddings.cube import open_cube
from eo_embeddings.tessera import open_reader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script

from pilot import common

PUBLISH_DIR = common.ROOT / "publish"


def raw_blocks_equal(a, b) -> bool:
    """True if both (embedding, scale) pairs match exactly, NaN equal to NaN."""
    return all(np.array_equal(x, y, equal_nan=x.dtype.kind == "f") for x, y in zip(a, b))


def no_data_counts(scales: np.ndarray, inside: np.ndarray) -> tuple[int, int]:
    """Among inside pixels: (failing valid_scale_mask, of those, finite scale)."""
    failing = inside & ~valid_scale_mask(scales)
    return int(failing.sum()), int((failing & np.isfinite(scales)).sum())


def cog_has_valid_pixels(path) -> bool:
    with rasterio.open(path) as src:
        return any(src.read_masks(i).any() for i in src.indexes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true", help="check the slice manifest and cube")
    args = ap.parse_args()
    manifest = common.ROOT / "pilot" / "slice.publish.yaml" if args.slice else (
        PUBLISH_DIR / "indre-et-loire-2025.publish.yaml"
    )
    cube_path = common.CUBE_DIR / ("slice.zarr" if args.slice else "2025.zarr")
    failed = False

    v = subprocess.run(
        ["platform-publish", "validate", str(PUBLISH_DIR / "project.yaml"), str(manifest)],
        check=False,
    )
    if v.returncode:
        print("FAIL: platform-publish validate")
        failed = True

    layers = yaml.safe_load(manifest.read_text())["layers"]
    for layer in layers:
        path = (manifest.parent / layer["file"]).resolve()
        if not path.exists():
            print(f"FAIL: {layer['slug']}: missing {path}")
            failed = True
        elif not cog_has_valid_pixels(path):
            print(f"FAIL: {layer['slug']}: no valid pixels")
            failed = True
        else:
            print(f"ok: {layer['slug']} has valid pixels")

    cube = open_cube(cube_path)
    a = cube.attrs
    bs = a["block_size"]
    n_rows, n_cols = -(-a["height"] // bs), -(-a["width"] // bs)
    row, col = n_rows // 2, n_cols // 2
    geometry = json.loads(common.BOUNDARY.read_text())["geometry"]
    box = common.SLICE_BOX if args.slice else geometry_bounds(geometry)
    reader = open_reader(box, common.YEAR, common.CACHE_DIR, block_size=bs)
    if (reader.grid.height, reader.grid.width) != (a["height"], a["width"]):
        print("FAIL: reader grid differs from the cube grid")
        failed = True
    else:
        same = raw_blocks_equal(cube.read_raw_block(row, col), reader.read_window(row, col))
        print(f"{'ok' if same else 'FAIL'}: block ({row}, {col}) matches the source")
        failed |= not same

    failing = finite = inside_total = 0
    for r, c, tr in cube.blocks():
        _, scales = cube.read_raw_block(r, c)
        inside = common.rasterize_boundary(geometry, tr, scales.shape, a["crs"])
        f, fin = no_data_counts(scales, inside)
        failing, finite, inside_total = failing + f, finite + fin, inside_total + int(inside.sum())
    print(f"inside pixels: {inside_total}")
    print(f"inside pixels failing valid_scale_mask: {failing}")
    print(f"  of those, with a finite scale (out of range): {finite}")
    if finite:
        print("WARNING: out-of-range finite scales exist in this store")

    print("FAILED" if failed else "PASSED")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
