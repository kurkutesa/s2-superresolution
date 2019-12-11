"""
This module includes necessary helper functions that are used in the s2_tiles_supres
and test_snap_polarimetry scripts.
"""
import logging
import os
import json
from pathlib import Path
from geojson import Feature, FeatureCollection
import rasterio
from rio_cogeo.profiles import cog_profiles

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SENTINEL2_L1C = "up42.data.scene.sentinel2_l1c"


def get_logger(name, level=logging.DEBUG):
    """
    This method creates logger object and sets the default log level to DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    c_h = logging.StreamHandler()
    c_h.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    c_h.setFormatter(formatter)
    logger.addHandler(c_h)

    return logger


def ensure_data_directories_exist():
    """
    This method checks input and output directories for data flow
    """
    Path('/tmp/input/').mkdir(parents=True, exist_ok=True)
    Path('/tmp/output/').mkdir(parents=True, exist_ok=True)


def load_metadata() -> FeatureCollection:
    """
    Get the data from the provided location
    """
    if os.path.exists("/tmp/input/data.json"):
        with open("/tmp/input/data.json") as f_p:
            data = json.loads(f_p.read())

        features = []
        for feature in data["features"]:
            features.append(Feature(**feature))

    return FeatureCollection(features)


def load_params() -> dict:
    """
    Get the parameters for the current task directly from the task parameters.
    """
    helper_logger = get_logger(__name__)
    data = os.environ.get("UP42_TASK_PARAMETERS", '{}')  # type: str
    helper_logger.debug("Fetching parameters for this block: %s", data)
    if data == "":
        data = "{}"
    return json.loads(data)


# pylint: disable-msg=too-many-arguments
def save_result(model_output, output_bands, valid_desc,
                output_profile, output_features, output_dir, image_name):
    """
    This method saves the feature collection meta data and the
    image with high resolution for desired bands to the provided location.
    :param model_output: The high resolution image.
    :param output_bands: The associated bands for the output image.
    :param valid_desc: The valid description of the existing bands.
    :param output_profile: The georeferencing for the output image.
    :param output_features: The meta data for the output image.
    :param output_dir: The directory in which the output image
        and associated meta data will be saved.
    :param image_name: The name of the output image.

    """
    # Use cloud-optimized geotiff for output
    cogeo_options = cog_profiles.get("deflate")
    out_profile = output_profile.copy()
    out_profile.update(cogeo_options)

    with rasterio.open(image_name, "w", **out_profile) as d_s:
        for b_i, b_n in enumerate(output_bands):
            d_s.write(model_output[:, :, b_i], indexes=b_i + 1)
            d_s.set_band_description(b_i+1, "SR " + valid_desc[b_n])

    with open(output_dir + "data.json", "w") as f_p:
        f_p.write(json.dumps(output_features, indent=2))
