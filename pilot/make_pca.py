"""Fit a 3-component PCA on the cube and write tessera-pca.tif and pca_model.json.

Run: uv run --group pilot python -m pilot.make_pca [--slice]
"""

import argparse
import json

import numpy as np
import rasterio
from geotessera.zarr import valid_scale_mask
from rasterio.transform import Affine
from rasterio.windows import Window

from eo_embeddings.cube import open_cube
from pilot import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true", help="use the slice cube and output")
    args = ap.parse_args()
    name = "slice" if args.slice else "2025"
    out_dir = common.SLICE_OUTPUT_DIR if args.slice else common.OUTPUT_DIR

    cube = open_cube(common.CUBE_DIR / f"{name}.zarr")  # refuses an incomplete cube
    a = cube.attrs
    geometry = json.loads(common.BOUNDARY.read_text())["geometry"]

    bs = a["block_size"]
    blocks = []  # (row, col, transform, inside) for Blocks overlapping the boundary
    for r, c, tr in cube.blocks():
        shape = (min(bs, a["height"] - r * bs), min(bs, a["width"] - c * bs))
        inside = common.rasterize_boundary(geometry, tr, shape, a["crs"])
        if inside.any():
            blocks.append((r, c, tr, inside))
    print(f"{len(blocks)} blocks overlap the boundary")

    def valid_pixels():
        for r, c, _, inside in blocks:
            emb, scales = cube.read_block(r, c)
            keep = inside & valid_scale_mask(scales)
            yield emb[keep]  # read_block gives (y, x, band)

    sample = common.sample_pixels(valid_pixels(), n_blocks=len(blocks))
    print(f"sampled {len(sample)} pixels")

    mean, components, variance, variance_ratio = common.fit_pca(sample)
    components, signs = common.apply_sign_rule(components)
    proj = common.project(sample, mean, components)
    p2, p98 = np.percentile(proj, [2, 98], axis=0)
    if (p98 == p2).any():
        raise ValueError(f"p98 equals p2 for a component: {p2} {p98}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tessera-pca.tmp.tif"
    profile = dict(
        driver="GTiff", count=3, height=a["height"], width=a["width"], dtype="uint8",
        crs=a["crs"], transform=Affine(*a["transform"]), nodata=0,
        tiled=True, blockxsize=256, blockysize=256, compress="DEFLATE",
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(tmp, "w", **profile) as dst:
        for r, c, _, inside in blocks:
            emb, scales = cube.read_block(r, c)
            out = common.pca_pixels(emb.transpose(2, 0, 1), scales, inside, mean, components, p2, p98)
            dst.write(out, window=Window(c * bs, r * bs, out.shape[2], out.shape[1]))
    common.to_cog(tmp, out_dir / "tessera-pca.tif", resampling="AVERAGE")

    (out_dir / "pca_model.json").write_text(json.dumps({
        "mean": mean.tolist(),
        "components": components.tolist(),
        "sign_rule": "each component's largest loading by magnitude is positive",
        "signs_applied": signs.tolist(),
        "seed": common.SEED,
        "sample_size": len(sample),
        "percentiles": {"p2": p2.tolist(), "p98": p98.tolist()},
        "explained_variance": variance.tolist(),
        "explained_variance_ratio": variance_ratio.tolist(),
        "store_url": a["store_url"],
        "geotessera_version": a["geotessera_version"],
    }, indent=2))
    print(f"wrote {out_dir / 'tessera-pca.tif'}")


if __name__ == "__main__":
    main()
