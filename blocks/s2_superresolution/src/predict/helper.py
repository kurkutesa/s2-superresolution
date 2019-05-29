import logging
import os
import json
from typing import Any, Union
from geojson import Feature, FeatureCollection
import rasterio

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
AOICLIPPED = "up42.data.scene.sentinel2_l1c"


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


def save_result(model_output, output_bands, output_profile, output_features, output_dir, image_name):

    with rasterio.open(image_name, "w", **output_profile) as ds:
        for bi, bn in enumerate(output_bands):
            ds.write(model_output[:, :, bi], indexes=bi + 1)

    with open(output_dir + "data.json", "w") as fp:
        fp.write(json.dumps(output_features, indent=2))
