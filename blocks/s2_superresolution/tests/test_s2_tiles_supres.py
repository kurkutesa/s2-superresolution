"""
This module include multiple test cases to check the performance of the s2_tiles_supres script.
"""
import glob
import os
import json
import rasterio
import pytest
import mock

from context import Superresolution, load_params, load_metadata, SENTINEL2_L1C

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
    input_metadata = load_metadata()
    img_id = input_metadata.features[0]["properties"][SENTINEL2_L1C]
    ds10r, ds20r, ds60r = fixture_1.get_data(img_id)
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
def fixture_2() -> list:
    """
    This method initiates the Superresolution class from s2_tiles_supres and apply the run
    method on it to produce an output image.
    """
    params = load_params()  # type: dict
    Superresolution(params).run()
    for out_file in glob.iglob(os.path.join('/tmp/output/', '*.tif'), recursive=True):
        output_image_path = out_file
    for json_out_file in glob.iglob(os.path.join('/tmp/output/', '*.json'), recursive=True):
        output_json_path = json_out_file
    return [output_image_path, output_json_path]


# pylint: disable=redefined-outer-name
def test_output_transform(fixture_2):
    """
    This method checks whether the outcome image has the correct 10m resolution for
    all the spectral bands.
    """
    output_image = rasterio.open(fixture_2[0])
    assert output_image.transform[0] == 10
    assert output_image.transform[4] == -10


# pylint: disable=redefined-outer-name
def test_output_description(fixture_2):
    """
    This method checks whether the outcome image has all the spectral bands for
    20m and 60m resolutions.
    """
    output_image = rasterio.open(fixture_2[0])
    desc_exm = ('SR B5 (705 nm)', 'SR B6 (740 nm)',
                'SR B7 (783 nm)', 'SR B8A (865 nm)', 'SR B11 (1610 nm)',
                'SR B12 (2190 nm)', 'SR B1 (443 nm)', 'SR B9 (945 nm)')
    assert output_image.descriptions == desc_exm


# pylint: disable=redefined-outer-name
def test_output_projection(fixture_1, fixture_2):
    """
    This method checks whether the outcome image has the correct georeference.
    """
    output_image = rasterio.open(fixture_2[0])
    input_metadata = load_metadata()
    img_id = input_metadata.features[0]["properties"][SENTINEL2_L1C]
    ds10r, ds20r, ds60r = fixture_1.get_data(img_id)
    crs_exm = fixture_1.get_utm(ds10r)
    assert output_image.crs == crs_exm


def test_output_json_file(fixture_2):
    """This method check whether the data.json of output image is created correctly."""
    with open(fixture_2[1]) as f_p:
        data = json.loads(f_p.read())

    assert 'features' in [*data]
