import pytest

import numpy as np

from context import patches

# pylint: disable=redefined-outer-name
@pytest.fixture()
def dset_10():
    return np.ones((10980, 10980, 4))


@pytest.fixture()
def dset_20():
    return np.ones((5490, 5490, 5))


@pytest.fixture()
def dset_60():
    return np.ones((1830, 1830, 3))


@pytest.fixture()
def scale_20():
    return 2


@pytest.fixture()
def scale_60():
    return 6


def test_get_test_patches(dset_10, dset_20):
    r = patches.get_test_patches(dset_10, dset_20, 128, 8)
    assert len(r) == 2
    assert r[0].shape == (9801, 4, 128, 128)
    assert r[1].shape == (9801, 5, 128, 128)


def test_get_test_patches60(dset_10, dset_20, dset_60):
    r = patches.get_test_patches60(dset_10, dset_20, dset_60, 192, 12)
    assert len(r) == 3
    assert r[0].shape == (4356, 4, 192, 192)
    assert r[1].shape == (4356, 5, 192, 192)
    assert r[2].shape == (4356, 3, 192, 192)


def test_get_crop_window():
    w = patches.get_crop_window(100, 50, 25)
    assert w == [100, 50, 125, 75]
    w = patches.get_crop_window(100, 50, 25, 2)
    assert w == [200, 100, 250, 150]


def test_crop_array_to_window():
    ar = np.ones(shape=(100, 100, 4))
    w = patches.get_crop_window(50, 50, 25)
    assert patches.crop_array_to_window(ar, w).shape == (4, 25, 25)
    assert patches.crop_array_to_window(ar, w, False).shape == (25, 25, 4)


def test_recompose_images(dset_10, dset_20):
    p = patches.get_test_patches(dset_10, dset_20, 128, 8)
    r_p = patches.recompose_images(p[0], 8, dset_10.shape)
    assert dset_10.shape == r_p.shape


def test_get_test_patches_wrong_number():
    dset_10 = np.ones((672, 606, 4))
    dset_20 = np.ones((335, 302, 6))
    dset_60 = np.ones((111, 100, 2))
    r_20 = patches.get_test_patches(dset_10, dset_20)
    r_60 = patches.get_test_patches60(dset_10, dset_20, dset_60)
    assert len(r_20) == 2
    assert r_20[0].shape == (36, 4, 128, 128)
    assert r_20[1].shape == (36, 6, 128, 128)

    assert len(r_60) == 3
    assert r_60[0].shape == (16, 4, 192, 192)
    assert r_60[1].shape == (16, 6, 192, 192)
    assert r_60[2].shape == (16, 2, 192, 192)
