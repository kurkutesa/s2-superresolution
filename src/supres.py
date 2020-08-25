from __future__ import division

import tensorflow as tf
from tensorflow import keras
from blockutils.logging import get_logger

from patches import get_test_patches, get_test_patches60, recompose_images

LOGGER = get_logger(__name__)
# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.

SCALE = 2000
MDL_PATH = "./weights/"

MDL_PATH_20M_AESR = MDL_PATH + "aesr_20m_s2_038_lr_1e-04.hdf5"
MDL_PATH_60M_AESR = MDL_PATH + "aesr_60m_s2_038_lr_1e-04.hdf5"

STRATEGY = tf.distribute.MirroredStrategy()


def dsen2_20(d10, d20):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 8
    p10, p20 = get_test_patches(d10, d20, patch_size=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = [p10, p20]
    prediction = _predict(test, model_filename=MDL_PATH_20M_AESR)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def dsen2_60(d10, d20, d60):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     d60: [x/6,y/6,2]  (B1, B9) -- NOT B10
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 12
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patch_size=192, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p60 /= SCALE
    test = [p10, p20, p60]
    prediction = _predict(test, model_filename=MDL_PATH_60M_AESR)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def _predict(test, model_filename):
    # create model
    with STRATEGY.scope():
        model = keras.models.load_model(model_filename)
    LOGGER.info("Symbolic Model Created.")
    LOGGER.info("Predicting using file: %s", model_filename)
    prediction = model.predict(test, verbose=1)
    return prediction
