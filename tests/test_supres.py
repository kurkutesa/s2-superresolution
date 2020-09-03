"""
This module include multiple test cases to check the performance of the s2_tiles_supres script.
"""
import tensorflow as tf
import numpy as np
import pytest
from context import dsen2_60, dsen2_20, BatchGenerator, patches

DISABLE_NO_GPU = pytest.mark.skipif(
    len(tf.config.list_physical_devices("GPU")) == 0,
    reason="Conv2D op requires GPU for channels first configuration.",
)

# pylint: disable=redefined-outer-name
@pytest.fixture
def level1():
    return "MSIL1C"


@pytest.fixture
def level2():
    return "MSIL2A"


@pytest.fixture()
def d10():
    return np.ones((10980 // 4, 10980 // 4, 4))


@pytest.fixture()
def d20():
    return np.ones((5490 // 4, 5490 // 4, 6))


@pytest.fixture()
def d60():
    return np.ones((1830 // 4, 1830 // 4, 2))


@DISABLE_NO_GPU
def test_dsen2_20(d10, d20, level1):
    res = dsen2_20(d10, d20, level1)
    assert res is not None
    assert res.shape[0:1] == d10.shape[0:1]


@DISABLE_NO_GPU
def test_dsen2_60(d10, d20, d60, level1):
    res = dsen2_60(d10, d20, d60, level1)
    assert res is not None
    assert res.shape[0:1] == d10.shape[0:1]


# @DISABLE_NO_GPU
def test_batch_generator(d10, d20):
    p10, p20 = patches.get_test_patches(d10, d20, 128, 8)
    a = BatchGenerator([p10, p20], batch_size=8)
    a_one = next(a)
    assert len(a_one) == 2
    assert a_one[0].shape == (9, 4, 128, 128)
    assert a_one[1].shape == (9, 6, 128, 128)

    a = BatchGenerator([p10, p20], batch_size=5000)
    a_one = next(a)
    assert len(a_one) == 2
    assert a_one[0].shape == (625, 4, 128, 128)
    assert a_one[1].shape == (625, 6, 128, 128)
