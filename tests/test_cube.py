import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr
from rasterio.transform import Affine

from eo_embeddings import (
    Cube,
    CubeIdentityError,
    Grid,
    IncompleteCubeError,
    open_cube,
    write_cube,
)

BANDS = 3
GRID = Grid(
    crs="EPSG:32631",
    transform=Affine(10, 0, 300000, 0, -10, 5270000),
    height=7,
    width=5,
    bands=BANDS,
    block_size=3,
    year=2025,
    store_url="https://example.org/v1.1",
    geotessera_version="0.11.0",
    mask_source="source_nodata",
)


def synth(grid, row, col):
    """Deterministic Block content, distinct per Block, shaped like a TesseraReader window."""
    bs = grid.block_size
    h = min(bs, grid.height - row * bs)
    w = min(bs, grid.width - col * bs)
    rng = np.random.default_rng(row * 100 + col)
    emb = rng.integers(-127, 128, size=(BANDS, h, w), dtype=np.int8)
    scales = rng.uniform(0.01, 0.9, size=(h, w)).astype(np.float32)
    return emb, scales


class Reader:
    def __init__(self, grid=GRID, fail_at=None):
        self.grid, self.fail_at, self.calls = grid, fail_at, []

    def __call__(self, row, col):
        if (row, col) == self.fail_at:
            raise RuntimeError("boom")
        self.calls.append((row, col))
        return synth(self.grid, row, col)


def journal_of(path):
    return Path(f"{path}.journal.jsonl")


def raw(path):
    g = zarr.open_group(str(path), mode="r")
    return g["embedding_int8"][:], g["scale"][:]


def test_stored_values_equal_input_and_layout(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    cube = open_cube(path)
    for r, c in GRID.blocks():
        emb, sc = cube.read_raw_block(r, c)
        e0, s0 = synth(GRID, r, c)
        assert emb.dtype == np.int8 and np.array_equal(emb, e0)
        assert np.array_equal(sc, s0)
    g = zarr.open_group(str(path), mode="r")
    assert g["embedding_int8"].shape == (1, BANDS, 7, 5)
    assert g["scale"].shape == (1, 7, 5)
    assert list(g["time"][:]) == [2025]
    assert g.attrs["status"] == "complete"
    assert cube.attrs["crs"] == "EPSG:32631"
    assert cube.attrs["source"] == "tessera"
    assert cube.attrs["format_version"] == 1


def test_unwritten_pixels_use_fill_values(tmp_path):
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 1)))
    g = zarr.open_group(str(path), mode="r")
    assert (g["embedding_int8"][0, :, 3:6, 3:5] == 0).all()
    assert np.isnan(g["scale"][0, 3:6, 3:5]).all()


def test_partial_edges_round_trip(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    cube = open_cube(path)
    emb, sc = cube.read_raw_block(2, 1)
    assert emb.shape == (BANDS, 1, 2) and sc.shape == (1, 2)
    assert np.array_equal(emb, synth(GRID, 2, 1)[0])


def test_blocks_yield_transforms(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    blocks = list(open_cube(path).blocks())
    assert [(r, c) for r, c, _ in blocks] == list(GRID.blocks())
    r, c, tr = blocks[-1]
    assert (r, c) == (2, 1)
    assert tuple(tr)[:6] == (10, 0, 300030, 0, -10, 5270000 - 60)


def test_resume_reads_only_unfinished_and_equals_clean_run(tmp_path):
    clean = tmp_path / "clean.zarr"
    write_cube(clean, GRID, Reader())
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 0)))
    assert not _is_complete(path)
    reader = Reader()
    write_cube(path, GRID, reader)
    assert reader.calls == [(1, 0), (1, 1), (2, 0), (2, 1)]
    for a, b in zip(raw(path), raw(clean)):
        assert np.array_equal(a, b, equal_nan=True)
    assert open_cube(path).attrs["status"] == "complete"


def _is_complete(path):
    return zarr.open_group(str(path), mode="r").attrs["status"] == "complete"


def test_torn_last_journal_line_rewrites_that_block(tmp_path):
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 0)))
    with journal_of(path).open("a") as f:
        f.write('{"row": 1, "co')
    reader = Reader()
    write_cube(path, GRID, reader)
    assert reader.calls[0] == (1, 0)
    assert len(reader.calls) == 4
    # journal is clean again: every line parses
    for line in journal_of(path).read_text().splitlines():
        json.loads(line)


def test_torn_earlier_journal_line_stops_the_run(tmp_path):
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 0)))
    lines = journal_of(path).read_text().splitlines()
    journal_of(path).write_text("garbage\n" + "\n".join(lines[1:]) + "\n")
    reader = Reader()
    with pytest.raises(ValueError):
        write_cube(path, GRID, reader)
    assert reader.calls == []


def test_rerun_after_complete_writes_nothing(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    before = journal_of(path).read_text()
    reader = Reader()
    write_cube(path, GRID, reader)
    assert reader.calls == []
    assert journal_of(path).read_text() == before


@pytest.mark.parametrize(
    "change",
    [
        {"store_url": "https://example.org/v2"},
        {"geotessera_version": "0.12.0"},
        {"crs": "EPSG:32632"},
        {"transform": Affine(10, 0, 300010, 0, -10, 5270000)},
        {"height": 8},
        {"block_size": 4},
        {"mask_source": "other"},
        {"year": 2024},
    ],
)
def test_resume_with_different_identity_is_refused(tmp_path, change):
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 0)))
    reader = Reader()
    with pytest.raises(CubeIdentityError):
        write_cube(path, replace(GRID, **change), reader)
    assert reader.calls == []


def test_identity_checked_on_complete_cube_too(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    with pytest.raises(CubeIdentityError):
        write_cube(path, replace(GRID, crs="EPSG:32632"), Reader())


def test_store_without_journal_is_refused(tmp_path):
    path = tmp_path / "c.zarr"
    write_cube(path, GRID, Reader())
    journal_of(path).unlink()
    with pytest.raises(CubeIdentityError):
        write_cube(path, GRID, Reader())


def test_journal_without_store_is_a_fresh_start(tmp_path):
    path = tmp_path / "c.zarr"
    journal_of(path).write_text('{"row": 0, "col": 0}\n')
    reader = Reader()
    write_cube(path, GRID, reader)
    assert len(reader.calls) == 6


def test_open_cube_refuses_incomplete(tmp_path):
    path = tmp_path / "c.zarr"
    with pytest.raises(RuntimeError):
        write_cube(path, GRID, Reader(fail_at=(1, 0)))
    with pytest.raises(IncompleteCubeError):
        open_cube(path)


def test_read_block_masks_invalid_scales_and_keeps_raw_scales(tmp_path):
    grid = replace(GRID, height=3, width=3)
    bad = np.array([[np.nan, np.inf, 0.0], [2.0, 0.5, -1.0], [0.25, 1.0, 0.1]], np.float32)
    emb = np.full((BANDS, 3, 3), 4, dtype=np.int8)
    path = tmp_path / "c.zarr"
    write_cube(path, grid, lambda r, c: (emb, bad))
    out, sc = open_cube(path).read_block(0, 0)
    assert out.dtype == np.float32 and out.shape == (3, 3, BANDS)
    valid = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], bool)  # 1.0 is not < MAX_VALID_SCALE
    assert np.isnan(out[~valid]).all()
    assert np.allclose(out[0 + 1, 1], 4 * 0.5)
    assert np.allclose(out[2, 0], 4 * 0.25)
    assert not np.isnan(out[valid]).any()
    assert np.array_equal(sc, bad, equal_nan=True)
