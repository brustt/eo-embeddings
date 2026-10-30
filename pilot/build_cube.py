"""Build the local Zarr cube: the 5 km slice with --slice, else the boundary's bounding box.

Run: uv run --group pilot python -m pilot.build_cube [--slice]
"""

import argparse
import json
import shutil
import time

from rasterio.features import bounds as geometry_bounds

from eo_embeddings.cube import write_cube
from eo_embeddings.tessera import open_reader
from pilot import common


def dir_size(path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true", help="build the 5 km test cube")
    args = ap.parse_args()

    # Chunks are not reused across runs, and the cache limit resets per process.
    shutil.rmtree(common.CACHE_DIR, ignore_errors=True)
    common.CACHE_DIR.mkdir(parents=True)

    if args.slice:
        box, path = common.SLICE_BOX, common.CUBE_DIR / "slice.zarr"
    else:
        geometry = json.loads(common.BOUNDARY.read_text())["geometry"]
        box, path = geometry_bounds(geometry), common.CUBE_DIR / "2025.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)

    reader = open_reader(box, common.YEAR, common.CACHE_DIR)
    g = reader.grid
    print(f"{g.crs} {g.height} x {g.width} px, {g.bands} bands, block {g.block_size}")
    start = time.monotonic()
    write_cube(path, g, reader.read_window)
    elapsed = time.monotonic() - start

    size = dir_size(path)
    pixels = g.height * g.width
    print(f"compressed size: {size / 1e6:.1f} MB ({size / pixels:.1f} bytes/pixel)")
    print(f"speed: {pixels / elapsed:.0f} pixels/s over {elapsed:.0f} s")
    print(f"cache size: {dir_size(common.CACHE_DIR) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
