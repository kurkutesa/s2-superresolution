from __future__ import division

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from blockutils.logging import get_logger

from patches import get_test_patches, get_test_patches60, recompose_images

LOGGER = get_logger(__name__)
# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.

SCALE = 2000
MDL_PATH = "./weights/"

L1C_MDL_PATH_20M_AESR = MDL_PATH + "l1c_aesr_20m_s2_038_lr_1e-04.hdf5"
L1C_MDL_PATH_60M_AESR = MDL_PATH + "l1c_aesr_60m_s2_038_lr_1e-04.hdf5"
L2A_MDL_PATH_20M_AESR = MDL_PATH + "l2a_aesr_20m_s2_038_lr_1e-04.hdf5"
L2A_MDL_PATH_60M_AESR = MDL_PATH + "l2a_aesr_60m_s2_038_lr_1e-04.hdf5"

STRATEGY = tf.distribute.MirroredStrategy()


def dsen2_20(d10, d20, image_level):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 8
    p10, p20 = get_test_patches(d10, d20, patch_size=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = [p10, p20]
    if image_level == "MSIL1C":
        model_filename = L1C_MDL_PATH_20M_AESR
    else:
        model_filename = L2A_MDL_PATH_20M_AESR

    prediction = _predict(test, model_filename)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def dsen2_60(d10, d20, d60, image_level):
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
    if image_level == "MSIL1C":
        model_filename = L1C_MDL_PATH_60M_AESR
    else:
        model_filename = L2A_MDL_PATH_60M_AESR
    prediction = _predict(test, model_filename)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


class BatchGenerator:
    def __init__(self, dataset_list, batch_size=128):
        self.batch_size = batch_size
        self.n_batches = dataset_list[0].shape[0] // batch_size
        if not self.n_batches:
            self.n_batches = 1
        LOGGER.info(f"Dividing into {self.n_batches} batches.")
        self.data_list_splitted = [
            np.array_split(d, self.n_batches, axis=0) for d in dataset_list
        ]
        self.len = len(self.data_list_splitted[0])
        LOGGER.info(f"Each batch has {self.data_list_splitted[0][0].shape[0]} patches.")
        self.iter = iter(zip(*self.data_list_splitted))

    def __len__(self):
        return self.len

    def __next__(self):
        return next(self.iter)

    def __iter__(self):
        return self


def _predict(test, model_filename):
    with STRATEGY.scope():
        model = keras.models.load_model(model_filename)
    LOGGER.info("Symbolic Model Created.")
    LOGGER.info("Predicting using file: %s", model_filename)
    first = True
    for a_slice in tqdm(BatchGenerator(test)):
        if first:
            first = False
            prediction = model.predict(a_slice)
        else:
            prediction = np.append(prediction, model.predict(a_slice), axis=0)

    LOGGER.info("Predicted...")
    del model
    return prediction
