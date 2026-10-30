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


# --- PCA pieces ---------------------------------------------------------------


def test_sign_rule_makes_each_largest_loading_positive():
    comps = np.array([[0.1, -0.9, 0.2], [0.5, 0.1, 0.3], [-0.2, -0.1, -0.8]])
    flipped, signs = common.apply_sign_rule(comps)
    assert signs.tolist() == [-1, 1, -1]
    assert flipped[0].tolist() == [-0.1, 0.9, -0.2]
    assert flipped[1].tolist() == comps[1].tolist()
    assert flipped[2].tolist() == [0.2, 0.1, 0.8]


def test_stretch_maps_percentiles_to_1_and_255_and_clips():
    v = np.array([-10.0, 0.0, 5.0, 10.0, 99.0])
    out = common.stretch(v, p2=0.0, p98=10.0)
    assert out.dtype == np.uint8
    assert out.tolist() == [1, 1, 128, 255, 255]  # rint(1 + 254 * 0.5) = 128


def test_stretch_stops_when_percentiles_are_equal():
    with pytest.raises(ValueError, match="p98"):
        common.stretch(np.array([1.0]), p2=3.0, p98=3.0)


def test_sample_is_reproducible_and_capped():
    rng_blocks = np.random.default_rng(42)
    blocks = [rng_blocks.normal(size=(n, 4)) for n in (50, 30, 80)]
    a = common.sample_pixels(blocks, total=60, seed=0)
    b = common.sample_pixels(blocks, total=60, seed=0)
    assert a.shape == (60, 4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, common.sample_pixels(blocks, total=60, seed=1))


def test_sample_takes_everything_when_fewer_than_total():
    blocks = [np.arange(8.0).reshape(2, 4), np.arange(4.0).reshape(1, 4)]
    out = common.sample_pixels(blocks, total=100, seed=0)
    assert out.shape == (3, 4)
    assert sorted(map(tuple, out)) == sorted(map(tuple, np.vstack(blocks)))


def test_sample_quota_is_ceil_total_over_blocks():
    # 3 blocks, total 10 -> quota ceil(10/3) = 4; the 2-pixel block gives all 2: pool of 10
    blocks = [np.ones((9, 2)), np.full((2, 2), 2.0), np.full((9, 2), 3.0)]
    out = common.sample_pixels(blocks, total=10, seed=0)
    assert out.shape == (10, 2)
    assert sorted(set(out[:, 0])) == [1.0, 2.0, 3.0]
    assert (out[:, 0] == 2.0).sum() == 2 and (out[:, 0] == 1.0).sum() == 4


def test_pca_pixels_follow_the_pixel_table():
    bands = 2
    emb = np.zeros((bands, 1, 5), dtype=np.float32)
    emb[0, 0, :] = [10, 10, 10, 10, 10]
    scales = np.array([[0.5, np.nan, np.inf, 0.0, 0.5]], dtype=np.float32)
    scales[0, 3] = 0.0  # no data
    inside = np.array([[True, True, True, True, False]])
    out = common.pca_pixels(
        emb, scales, inside,
        mean=np.zeros(bands), components=np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]]),
        p2=np.zeros(3), p98=np.full(3, 20.0),
    )
    assert out.shape == (3, 1, 5) and out.dtype == np.uint8
    assert out[:, 0, 0].tolist() == [128, 1, 128]  # valid, inside: stretched
    assert (out[:, 0, 1:] == 0).all()  # NaN, +inf, 0 scale, and outside: all 0


def test_scale_of_one_or_more_is_no_data():
    emb = np.full((1, 1, 2), 10, dtype=np.float32)
    scales = np.array([[1.0, 2.0]], dtype=np.float32)
    out = common.pca_pixels(
        emb, scales, np.ones((1, 2), bool),
        mean=np.zeros(1), components=np.ones((3, 1)), p2=np.zeros(3), p98=np.full(3, 20.0),
    )
    assert (out == 0).all()


def test_write_cog_utm_with_average_overviews(tmp_path):
    data = np.random.default_rng(0).integers(1, 255, (3, 600, 700), dtype=np.uint8)
    path = tmp_path / "x.tif"
    common.write_cog(
        path, data, Affine(10, 0, 500000, 0, -10, 5250000),
        crs="EPSG:32631", resampling="AVERAGE",
    )
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32631 and src.nodata == 0 and src.overviews(1)
