import numpy as np

from pilot import check_outputs as co


def test_raw_blocks_equal_treats_nan_as_equal():
    emb = np.zeros((2, 3, 3), dtype=np.int8)
    scales = np.array([[1.0, np.nan, 2.0]] * 3, dtype=np.float32)
    assert co.raw_blocks_equal((emb, scales), (emb.copy(), scales.copy()))


def test_raw_blocks_equal_fails_on_any_difference():
    emb = np.zeros((2, 3, 3), dtype=np.int8)
    scales = np.ones((3, 3), dtype=np.float32)
    other = emb.copy()
    other[1, 2, 2] = 1
    assert not co.raw_blocks_equal((emb, scales), (other, scales))
    nan_scales = scales.copy()
    nan_scales[0, 0] = np.nan
    assert not co.raw_blocks_equal((emb, scales), (emb, nan_scales))


def test_no_data_counts_separate_nan_from_finite_out_of_range():
    # valid, NaN (no data), negative (finite but out of range), valid-but-outside
    scales = np.array([[0.01, np.nan], [-3.0, 0.02]], dtype=np.float32)
    inside = np.array([[True, True], [True, False]])
    assert co.no_data_counts(scales, inside) == (2, 1)
