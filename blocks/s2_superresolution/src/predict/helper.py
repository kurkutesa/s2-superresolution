import logging
import os
import json
from typing import Any, Union
from geojson import Feature, FeatureCollection
import rasterio

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SENTINEL2_L1C = "up42.data.scene.sentinel2_l1c"


def get_logger(name, level=logging.DEBUG):
    """
    This method creates logger object and sets the default log level to DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_metadata() -> FeatureCollection:
    """
    Get the data from the provided location
    """
    if os.path.exists("/tmp/input/data.json"):
        with open("/tmp/input/data.json") as fp:
            data = json.loads(fp.read())

        features = []
        for feature in data["features"]:
            features.append(Feature(**feature))

        return FeatureCollection(features)
    else:
        return FeatureCollection([])


def load_params() -> dict:
    """
    Get the parameters for the current task directly from the task parameters. If
    the task parameters are not set, falls back to the old INTERSTELLAR_JOB_INPUTS.
    """
    helper_logger = get_logger(__name__)
    data = os.environ.get("UP42_TASK_PARAMETERS", '{}')
    helper_logger.debug("Fetching parameters for this blocks: %s", data)
    if data == "":
        data = "{}"
    return json.loads(data)


def save_result(model_output, output_bands, valid_desc, output_profile, output_features, output_dir, image_name):
    """
    This method saves the feature collection meta data and the image with high resolution for desired bands
    to the provided location.
    :param model_output: The high resolution image.
    :param output_bands: The associated bands for the output image.
    :param valid_desc: The valid description of the existing bands.
    :param output_profile: The georeferencing for the output image.
    :param output_features: The meta data for the output image.
    :param output_dir: The directory in which the output image and associated meta data will be saved.
    :param image_name: The name of the output image.

    """
    with rasterio.open(image_name, "w", **output_profile) as ds:
        for bi, bn in enumerate(output_bands):
            ds.write(model_output[:, :, bi], indexes=bi + 1)
            ds.set_band_description(bi+1, "SR " + valid_desc[bn])

    with open(output_dir + "data.json", "w") as fp:
        fp.write(json.dumps(output_features, indent=2))
