"""
This module creates the super-resolution image based on the trained CNN defined in dsen2Net script.
"""
import gc

import numpy as np
import keras.backend as K
import tensorflow as tf

from dsen2net import s2model
from patches import get_test_patches, get_test_patches60, recompose_images
from helper import get_logger


SCALE = 2000
MDL_PATH = "./weights/"
LOGGER = get_logger(__name__)
# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.


def dsen2_20(d10, d20, deep=False) -> np.ndarray:
    """
    This methods uses a convolutional neural network
    created from the _predict method to obtain a 10m resolution
    image from a 20m resolution.

    :param d10: The numpy array of pixels's value for 10m resolution.
    :param d20: The numpy array of pixels's value for 20m resolution.
    :param deep: If true, 32 would be the number of resBlocks used in the model and 256
    would be the number of filters used to scan the image to create feature maps.
    :return: The predicted high resolution (10 m) image.
    """
    border = 8
    p10, p20 = get_test_patches(d10, d20, patchsize=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = [p10, p20]
    input_shape = ((4, None, None), (6, None, None))
    prediction = _predict(test, input_shape, deep=deep)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def dsen2_60(d10, d20, d60, deep=False) -> np.ndarray:
    """
    This methods uses a convolutional neural network
    created from the _predict method to 10m resolution
    for 20m and 60m spectral bands.

    :param d10: The numpy array of pixels's value for 10m resolution.
    :param d20: The numpy array of pixels's value for 20m resolution.
    :param d60: The numpy array of pixels's value for 60m resolution.
    :param deep: If true, 32 would be the number of resBlocks used in the model and 256
    would be the number of filters used to scan the image to create feature maps.
    :return: The predicted high resolution (10 m) image.
    """
    border = 12
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patchsize=192, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p60 /= SCALE
    test = [p10, p20, p60]
    input_shape = ((4, None, None), (6, None, None), (2, None, None))
    prediction = _predict(test, input_shape, deep=deep, run_60=True)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def _predict(test, input_shape, deep=False, run_60=False):
    # create model
    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True  # pylint: disable=no-member

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = (
        0.5  # pylint: disable=no-member
    )

    # Terminate on long hangs
    # config.operation_timeout_in_ms = 15000

    # Create a session with the above options specified.
    session = tf.Session(config=config)
    K.set_session(session)
    ###################################
    if deep:
        model = s2model(input_shape, num_layers=32, feature_size=256)
        predict_file = (
            MDL_PATH + "s2_034_lr_1e-04.hdf5"
            if run_60
            else MDL_PATH + "s2_033_lr_1e-04.hdf5"
        )
    else:
        model = s2model(input_shape, num_layers=6, feature_size=128)
        predict_file = (
            MDL_PATH + "s2_030_lr_1e-05.hdf5"
            if run_60
            else MDL_PATH + "s2_032_lr_1e-04.hdf5"
        )
    LOGGER.info("Symbolic Model Created.")
    model.load_weights(predict_file)
    LOGGER.info("Predicting using file: %s", predict_file)
    prediction = model.predict(test, verbose=1)
    del model
    LOGGER.info("This is for releasing memory: %s", gc.collect())
    # finally, close sessions
    session.close()
    K.clear_session()
    return prediction