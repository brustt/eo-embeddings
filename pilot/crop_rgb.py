"""Crop the store's global_rgb to Indre-et-Loire as tessera-rgb.tif.

Run: uv run --group pilot python -m pilot.crop_rgb
"""

import json

import numpy as np
import zarr
from geotessera.registry import zarr_store_url
from rasterio.features import bounds as geometry_bounds
from rasterio.transform import Affine

from pilot import common


def main() -> None:
    geometry = json.loads(common.BOUNDARY.read_text())["geometry"]
    root = zarr_store_url("v1.1", "cambridge")
    group = zarr.open_group(root + "/global_rgb", mode="r")
    arr = zarr.open_array(root + "/global_rgb/0/rgb", mode="r")
    layout0 = group.attrs["multiscales"]["layout"][0]
    full = common.check_layout(arr.metadata.dimension_names, layout0, arr.shape)

    px = full.a
    r0, r1, c0, c1 = common.pixel_window(
        geometry_bounds(geometry), px, (full.c, full.f)
    )
    transform = full * Affine.translation(c0, r0)
    h, w = r1 - r0, c1 - c0
    print(f"window rows {r0}:{r1} cols {c0}:{c1} ({h} x {w})")

    out = np.zeros((3, h, w), dtype=np.uint8)
    inside_all = np.zeros((h, w), dtype=bool)
    alpha_all = np.zeros((h, w), dtype=bool)
    cr, cc = arr.chunks[:2]
    # Read by the array's own chunks.
    for gr in range(r0 // cr * cr, r1, cr):
        for gc in range(c0 // cc * cc, c1, cc):
            ra, rb = max(gr, r0), min(gr + cr, r1)
            ca, cb = max(gc, c0), min(gc + cc, c1)
            rgba = arr[ra:rb, ca:cb, :]
            tr = full * Affine.translation(ca, ra)
            inside = common.rasterize_boundary(
                geometry, tr, rgba.shape[:2], "EPSG:4326"
            )
            sl = (slice(ra - r0, rb - r0), slice(ca - c0, cb - c0))
            out[(slice(None), *sl)] = common.crop_rgb(rgba, inside)
            inside_all[sl] = inside
            alpha_all[sl] = rgba[..., 3] > 0

    counted = inside_all & alpha_all
    all_zero, partial = common.count_zero_pixels(out, counted)
    print(f"inside pixels with alpha > 0: {int(counted.sum())}")
    print(f"  of which (0, 0, 0): {all_zero}")
    print(f"  of which 0 in only some bands: {partial}")

    common.write_cog(common.RGB_COG, out, transform)
    print(f"wrote {common.RGB_COG}")


if __name__ == "__main__":
    main()
