"""Write, resume and read a local Zarr cube of quantized Tessera Blocks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import zarr
from geotessera.store import TesseraAccessor
from geotessera.zarr import valid_scale_mask
from rasterio.transform import Affine
from zarr.codecs import ZstdCodec

from eo_embeddings.tessera import Grid

FORMAT_VERSION = 1
SOURCE = "tessera"
CHUNK = 512


class CubeIdentityError(ValueError):
    """The stored cube was made with a different identity than the one requested."""


class IncompleteCubeError(ValueError):
    """The cube's status is not `complete`."""


def _journal_path(path) -> Path:
    return Path(f"{path}.journal.jsonl")


def _identity(grid: Grid) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "source": SOURCE,
        "store_url": grid.store_url,
        "geotessera_version": grid.geotessera_version,
        "mask_source": grid.mask_source,
        "crs": grid.crs,
        "transform": [float(v) for v in tuple(grid.transform)[:6]],
        "year": grid.year,
        "bands": grid.bands,
        "block_size": grid.block_size,
        "height": grid.height,
        "width": grid.width,
    }


def _create_store(path, grid: Grid, identity: dict) -> zarr.Group:
    codec = ZstdCodec(level=3)
    g = zarr.create_group(str(path), zarr_format=3, attributes={**identity, "status": "writing"})
    g.create_array(
        "embedding_int8",
        shape=(1, grid.bands, grid.height, grid.width),
        chunks=(1, grid.bands, CHUNK, CHUNK),
        dtype="int8",
        fill_value=0,
        compressors=codec,
        dimension_names=("time", "band", "y", "x"),
    )
    g.create_array(
        "scale",
        shape=(1, grid.height, grid.width),
        chunks=(1, CHUNK, CHUNK),
        dtype="float32",
        fill_value=float("nan"),
        compressors=codec,
        dimension_names=("time", "y", "x"),
    )
    t = g.create_array("time", shape=(1,), dtype="int32", dimension_names=("time",))
    t[:] = [grid.year]
    return g


def _read_journal(journal: Path) -> set[tuple[int, int]]:
    """Finished Blocks. A torn last line is dropped (and the file cleaned); any other bad line raises."""
    lines = journal.read_text().splitlines()
    done, good = set(), []
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
            key = (int(rec["row"]), int(rec["col"]))
        except (ValueError, KeyError, TypeError):
            if i == len(lines) - 1:
                break
            raise ValueError(f"Unparseable journal line {i + 1} in {journal}") from None
        done.add(key)
        good.append(line)
    if len(good) != len(lines) or (lines and not journal.read_text().endswith("\n")):
        journal.write_text("".join(f"{ln}\n" for ln in good))
    return done


def _append(journal: Path, row: int, col: int) -> None:
    with journal.open("a") as f:
        f.write(json.dumps({"row": row, "col": col}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_cube(path, grid: Grid, read_window: Callable[[int, int], tuple]) -> None:
    path = Path(path)
    journal = _journal_path(path)
    identity = _identity(grid)
    if path.exists():
        if not journal.exists():
            raise CubeIdentityError(f"Store {path} has no journal; refusing to resume")
        g = zarr.open_group(str(path), mode="r+")
        diff = [k for k, v in identity.items() if g.attrs.get(k) != v]
        if diff:
            raise CubeIdentityError(f"Stored cube differs from the requested one in: {diff}")
        if g.attrs["status"] == "complete":
            return
        done = _read_journal(journal)
    else:
        journal.write_text("")  # journal first, so a crash never leaves a store without one
        g = _create_store(path, grid, identity)
        done = set()
    emb_arr, scale_arr = g["embedding_int8"], g["scale"]
    bs = grid.block_size
    for row, col in grid.blocks():
        if (row, col) in done:
            continue
        emb, scales = read_window(row, col)
        r0, c0 = row * bs, col * bs
        emb_arr[0, :, r0 : r0 + emb.shape[1], c0 : c0 + emb.shape[2]] = emb
        scale_arr[0, r0 : r0 + scales.shape[0], c0 : c0 + scales.shape[1]] = scales
        _append(journal, row, col)
    g.attrs["status"] = "complete"


class Cube:
    def __init__(self, group: zarr.Group):
        self._g = group
        self.attrs = dict(group.attrs)

    def blocks(self) -> Iterator[tuple[int, int, Affine]]:
        a = self.attrs
        base, bs = Affine(*a["transform"]), a["block_size"]
        n_rows, n_cols = -(-a["height"] // bs), -(-a["width"] // bs)
        for r in range(n_rows):
            for c in range(n_cols):
                yield r, c, base @ Affine.translation(c * bs, r * bs)

    def _slices(self, row: int, col: int):
        a, bs = self.attrs, self.attrs["block_size"]
        r0, c0 = row * bs, col * bs
        return slice(r0, min(r0 + bs, a["height"])), slice(c0, min(c0 + bs, a["width"]))

    def read_raw_block(self, row: int, col: int):
        ys, xs = self._slices(row, col)
        return self._g["embedding_int8"][0, :, ys, xs], self._g["scale"][0, ys, xs]

    def read_block(self, row: int, col: int):
        emb, scales = self.read_raw_block(row, col)
        out = TesseraAccessor.dequantise(emb, scales)
        out[~valid_scale_mask(scales)] = np.nan
        return out, scales


def open_cube(path) -> Cube:
    g = zarr.open_group(str(path), mode="r")
    if g.attrs.get("status") != "complete":
        raise IncompleteCubeError(f"Cube {path} has status {g.attrs.get('status')!r}")
    return Cube(g)
