import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from pilot import common


def test_outside_and_alpha_zero_become_black_others_keep_values():
    rgba = np.array(
        [[[10, 20, 30, 255], [11, 21, 31, 0]], [[12, 22, 32, 255], [13, 23, 33, 128]]],
        dtype=np.uint8,
    )
    inside = np.array([[True, True], [False, True]])
    out = common.crop_rgb(rgba, inside)
    assert out.shape == (3, 2, 2) and out.dtype == np.uint8
    assert out[:, 0, 0].tolist() == [10, 20, 30]  # kept
    assert out[:, 0, 1].tolist() == [0, 0, 0]  # alpha 0
    assert out[:, 1, 0].tolist() == [0, 0, 0]  # outside
    assert out[:, 1, 1].tolist() == [13, 23, 33]  # alpha > 0 keeps value


def test_pixel_window_floors_west_north_and_ceils_east_south():
    # grid origin (-180, 90), 0.0001 degree pixels
    r0, r1, c0, c1 = common.pixel_window(
        (0.65605, 47.37301, 0.72205, 47.41805), px=0.0001, origin=(-180.0, 90.0)
    )
    assert (c0, c1) == (1806560, 1807221)
    assert (r0, r1) == (425819, 426270)


def test_rasterize_uses_pixel_centres():
    # square covering x in [0, 2.4], y in [0, 2.4]; 1-degree pixels, origin (0, 4)
    square = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [2.4, 0], [2.4, 2.4], [0, 2.4], [0, 0]]],
    }
    tr = Affine(1, 0, 0, 0, -1, 4)
    mask = common.rasterize_boundary(square, tr, (4, 4), "EPSG:4326")
    # centres at 0.5 and 1.5 are inside; 2.5 is outside
    assert mask[2:, :2].all()
    assert mask.sum() == 4


def test_count_zero_pixels_splits_all_zero_from_partial():
    rgb = np.zeros((3, 2, 2), dtype=np.uint8)
    rgb[:, 0, 0] = (0, 0, 0)  # all zero
    rgb[:, 0, 1] = (0, 5, 5)  # partial
    rgb[:, 1, 0] = (7, 7, 7)  # none
    rgb[:, 1, 1] = (0, 0, 0)  # all zero but not counted (mask excludes it)
    counted = np.array([[True, True], [True, False]])
    assert common.count_zero_pixels(rgb, counted) == (1, 1)


def test_write_cog_settings(tmp_path):
    data = np.random.default_rng(0).integers(1, 255, (3, 600, 700), dtype=np.uint8)
    path = tmp_path / "x.tif"
    common.write_cog(path, data, Affine(0.0001, 0, 0, 0, -0.0001, 1))
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8"
        assert src.nodata == 0 and src.crs.to_epsg() == 4326
        assert src.block_shapes[0] == (256, 256)
        assert src.overviews(1)
        assert src.compression.name == "deflate"
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        np.testing.assert_array_equal(src.read(), data)
        # nearest: every overview value is a source value
        ov = src.read(
            1,
            out_shape=(
                src.height // src.overviews(1)[0],
                src.width // src.overviews(1)[0],
            ),
        )
        assert set(np.unique(ov)) <= set(np.unique(data[0]))


def test_check_layout_accepts_store_facts_and_rejects_changes():
    attrs = ("lat", "lon", "band")
    layout = {"spatial:transform": [0.0001, 0, -180, 0, -0.0001, 90]}
    common.check_layout(attrs, layout, (10, 10, 4))
    for bad in (
        (("lon", "lat", "band"), layout, (10, 10, 4)),
        (attrs, {"spatial:transform": [0.0002, 0, -180, 0, -0.0002, 90]}, (10, 10, 4)),
        (attrs, layout, (10, 10, 3)),
    ):
        with pytest.raises(ValueError):
            common.check_layout(*bad)
