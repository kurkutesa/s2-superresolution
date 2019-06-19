"""
This module include multiple test cases to check the performance of the s2_tiles_supres script.
"""
import glob
import os
import rasterio
import pytest
import mock

from context import Superresolution

INPUT_FOLDER = '/tmp/input/'
DATA_FOLDER = '*/*/MTD*.xml'
for file in glob.iglob(os.path.join(INPUT_FOLDER, DATA_FOLDER), recursive=True):
    DATA_PATH = file


@pytest.fixture(scope="session", autouse=True)
def fixture_1():
    """
    This method initiates the Superresolution class from s2_tiles_supres to be
    used to testing.
    """
    params = {'roi_x_y': [5000, 5000, 5500, 5500]}
    return Superresolution(params)


# pylint: disable-msg=too-many-locals
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
def test_desc(fixture_1):
    """
    This method check whether validate function defined in the s2_tiles_supres
    file produce the correct results.
    """
    validated_10m_indices_exm = [0, 1, 2, 3]
    validated_10m_bands_exm = ['B2', 'B3', 'B4', 'B8']
    validated_20m_indices_exm = [0, 1, 2, 3, 4, 5]
    validated_20m_bands_exm = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    validated_60m_indices_exm = [0, 1]
    validated_60m_bands_exm = ['B1', 'B9']
    ds10r, ds20r, ds60r, output_jsonfile, output_name = fixture_1.get_data()
    validated_10m_bands, validated_10m_indices, dic_10m = fixture_1.validate(ds10r)
    validated_20m_bands, validated_20m_indices, dic_20m = fixture_1.validate(ds20r)
    validated_60m_bands, validated_60m_indices, dic_60m = fixture_1.validate(ds60r)
    assert set(validated_10m_bands) == set(validated_10m_bands_exm)
    assert set(validated_20m_bands) == set(validated_20m_bands_exm)
    assert set(validated_60m_bands) == set(validated_60m_bands_exm)
    assert validated_10m_indices == validated_10m_indices_exm
    assert validated_20m_indices == validated_20m_indices_exm
    assert validated_60m_indices == validated_60m_indices_exm


@pytest.fixture(scope="session", autouse=True)
@mock.patch.dict(os.environ, {"UP42_TASK_PARAMETERS": '{"roi_x_y": [5000, 5000, 5500, 5500]}'})
def fixture_2():
    """
    This method initiates the Superresolution class from s2_tiles_supres and apply the run
    method on it to produce an output image.
    """
    Superresolution.run()
    for out_file in glob.iglob(os.path.join('/tmp/output/', '*.tif'), recursive=True):
        output_image_path = out_file
    return rasterio.open(output_image_path)


# pylint: disable=redefined-outer-name
def test_output_transform(fixture_2):
    """
    This method checks whether the outcome image has the correct 10m resolution for
    all the spectral bands.
    """
    assert fixture_2.transform[0] == 10
    assert fixture_2.transform[4] == -10


# pylint: disable=redefined-outer-name
def test_output_description(fixture_2):
    """
    This method checks whether the outcome image has all the spectral bands for
    20m and 60m resolutions.
    """
    desc_exm = ('SR B5 (705 nm)', 'SR B6 (740 nm)',
                'SR B7 (783 nm)', 'SR B8A (865 nm)', 'SR B11 (1610 nm)',
                'SR B12 (2190 nm)', 'SR B1 (443 nm)', 'SR B9 (945 nm)')
    assert fixture_2.descriptions == desc_exm


# pylint: disable=redefined-outer-name
def test_output_projection(fixture_1, fixture_2):
    """
    This method checks whether the outcome image has the correct georeference.
    """
    ds10r, ds20r, ds60r, output_jsonfile, output_name = fixture_1.get_data()
    crs_exm = fixture_1.get_utm(ds10r)
    assert fixture_2.crs == crs_exm
