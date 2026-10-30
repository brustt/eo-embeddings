import numpy as np
import pytest
import xarray as xr
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform

from eo_embeddings import (
    Grid,
    MaskSourceError,
    TesseraReader,
    UtmZoneError,
    check_mask_source,
    open_reader,
    tessera,
)

BANDS = 4
XS = 300005.0 + 10.0 * np.arange(6000)  # pixel centres, 300000..360000
YS = 5270000.0 - 5.0 - 10.0 * np.arange(4000)  # descending, 5230000..5270000
TOURS = (0.6, 47.3, 0.9, 47.5)  # lon/lat, zone 31


class FakeAccessor:
    pixel_size = 10.0

    def __init__(self, ds):
        self.ds = ds
        self.calls = []
        self.shape_delta = (0, 0)
        self.shift = 0.0

    def read_region_quantized(self, bbox, year):
        self.calls.append((bbox, year))
        w, s, e, n = bbox
        xs, ys = self.ds.xs, self.ds.ys
        x0, x1 = np.searchsorted(xs, w, "left"), np.searchsorted(xs, e, "right")
        y0, y1 = np.searchsorted(-ys, -n, "left"), np.searchsorted(-ys, -s, "right")
        h, wd = y1 - y0 + self.shape_delta[0], x1 - x0 + self.shape_delta[1]
        emb = np.ones((h, wd, BANDS), dtype=np.int8)
        scales = np.ones((h, wd), dtype=np.float32)
        tr = Affine(10, 0, xs[x0] - 5 + self.shift, 0, -10, ys[y0] + 5)
        return emb, scales, tr


class FakeZone:
    def __init__(self, attrs=None):
        self.attrs = {"geotessera:mask_source": "source_nodata"} if attrs is None else attrs
        self.xs, self.ys = XS, YS
        self.tessera = FakeAccessor(self)
        self._emb = xr.DataArray(
            np.zeros((1, BANDS, 1, 1), dtype=np.int8), dims=("time", "band", "y", "x")
        )

    def __getitem__(self, key):
        if key == "x":
            return xr.DataArray(self.xs)
        if key == "y":
            return xr.DataArray(self.ys)
        return self._emb


@pytest.fixture
def zone(monkeypatch):
    z = FakeZone()
    opened = {}

    class FakeStore:
        def __init__(self, url, cache_dir=None, cache_max_size=None):
            opened.update(url=url, cache_dir=cache_dir, cache_max_size=cache_max_size)

        def open_zone(self, *, zone):
            opened["zone"] = zone
            return z

    monkeypatch.setattr(tessera, "GeoTesseraZarr", FakeStore)
    z.opened = opened
    return z


def test_box_across_utm_zones_raises(zone):
    with pytest.raises(UtmZoneError):
        open_reader((-0.5, 47.3, 0.5, 47.5), 2025, "c")


def test_box_straddling_zero_degrees_raises_but_east_of_it_does_not(zone):
    with pytest.raises(UtmZoneError):
        open_reader((-1e-6, 47.3, 0.5, 47.5), 2025, "c")
    # the department's west edge (~0.05 E) is in zone 31
    open_reader((0.0, 47.3, 0.5, 47.5), 2025, "c")
    assert zone.opened["zone"] == 31


@pytest.mark.parametrize("attrs", [{}, {"geotessera:mask_source": "landmask"}])
def test_check_mask_source_rejects_absent_or_other(attrs):
    with pytest.raises(MaskSourceError):
        check_mask_source(attrs)


def test_check_mask_source_returns_value():
    assert check_mask_source({"geotessera:mask_source": "source_nodata"}) == "source_nodata"


def test_open_reader_refuses_untrusted_store(monkeypatch, zone):
    zone.attrs = {}
    with pytest.raises(MaskSourceError):
        open_reader(TOURS, 2025, "c")


def test_open_reader_uses_cache_and_default_store(zone, tmp_path):
    open_reader(TOURS, 2025, tmp_path)
    assert zone.opened["cache_dir"] == tmp_path
    assert zone.opened["cache_max_size"] == 2 * 1024**3
    assert "v1.1-dclimate" in zone.opened["url"]


def test_grid_contains_densified_input_boundary(zone):
    grid = open_reader(TOURS, 2025, "c").grid
    w, s, e, n = TOURS
    t = np.linspace(0, 1, 50)
    lons = np.concatenate([w + t * (e - w), w + t * (e - w), np.full(50, w), np.full(50, e)])
    lats = np.concatenate([np.full(50, s), np.full(50, n), s + t * (n - s), s + t * (n - s)])
    xs, ys = warp_transform("EPSG:4326", "EPSG:32631", lons.tolist(), lats.tolist())
    left, top = grid.transform * (0, 0)
    right, bottom = grid.transform * (grid.width, grid.height)
    assert left <= min(xs) and max(xs) <= right
    assert bottom <= min(ys) and max(ys) <= top
    # snapped to the zone's pixel edges, not wildly larger
    assert (left - 300000) % 10 == 0
    assert left > min(xs) - 10 and right < max(xs) + 10
    assert grid.bands == BANDS
    assert grid.year == 2025
    assert str(grid.crs) == "EPSG:32631"
    assert grid.mask_source == "source_nodata"


def make_reader(zone, block_size=512, size=(1100, 1030)):
    grid = Grid(
        crs="EPSG:32631",
        transform=Affine(10, 0, XS[0] - 5, 0, -10, YS[0] + 5),
        height=size[0],
        width=size[1],
        bands=BANDS,
        block_size=block_size,
        year=2025,
        store_url="u",
        geotessera_version="0.11.0",
        mask_source="source_nodata",
    )
    return TesseraReader(grid, zone)


def test_blocks_include_partial_last_row_and_column(zone):
    grid = make_reader(zone).grid
    assert list(grid.blocks()) == [(r, c) for r in range(3) for c in range(3)]


def test_partial_windows_have_the_right_size(zone):
    reader = make_reader(zone)  # 1100 x 1030 -> last row 76, last col 6
    emb, scales = reader.read_window(0, 0)
    assert emb.shape == (BANDS, 512, 512) and scales.shape == (512, 512)
    assert emb.dtype == np.int8
    emb, scales = reader.read_window(2, 2)
    assert emb.shape == (BANDS, 76, 6) and scales.shape == (76, 6)
    emb, _ = reader.read_window(1, 2)
    assert emb.shape == (BANDS, 512, 6)


def test_read_window_passes_pixel_centres_not_edges(zone):
    reader = make_reader(zone)
    reader.read_window(1, 1)
    (w, s, e, n), year = zone.tessera.calls[0]
    assert year == 2025
    # block (1,1): rows 512..1023, cols 512..1023
    assert w == XS[512] and e == XS[1023]
    assert n == YS[512] and s == YS[1023]


def test_read_window_raises_on_shape_mismatch(zone):
    reader = make_reader(zone)
    zone.tessera.shape_delta = (1, 0)
    with pytest.raises(ValueError, match="shape"):
        reader.read_window(0, 0)


def test_read_window_raises_on_transform_mismatch(zone):
    reader = make_reader(zone)
    zone.tessera.shift = 10.0
    with pytest.raises(ValueError, match="transform"):
        reader.read_window(0, 0)
