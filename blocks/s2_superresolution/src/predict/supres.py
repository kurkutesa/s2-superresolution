import sys
import os
sys.path.append('../src')
from utils.DSen2Net import s2model
from utils.patches import get_test_patches, get_test_patches60, recompose_images


SCALE = 2000
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
MDL_PATH = 'weights/'


def DSen2_20(d10, d20, deep=False):
    """
    This methods uses a convolutional neural network created from the _predict method to obtain a 10m resolution
    image from a 20m resolution.

    :param d10: The numpy array of pixels's value for 10m resolution.
    :param d20: The numpy array of pixels's value for 20m resolution.
    :param deep: If true, 32 would be the number of resBlocks used in the model and 256
    would be the number of filters used to scan the image to create feature maps.
    :return: The predicted high resolution (10 m) image.
    """
    border = 8
    p10, p20 = get_test_patches(d10, d20, patchSize=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = [p10, p20]
    input_shape = ((4, None, None), (6, None, None))
    prediction = _predict(test, input_shape, deep=deep)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def DSen2_60(d10, d20, d60, deep=False):
    """
    This methods uses a convolutional neural network created from the _predict method to obtain a 10m resolution
    image based on 20m and 60m resolution images.

    :param d10: The numpy array of pixels's value for 10m resolution.
    :param d20: The numpy array of pixels's value for 20m resolution.
    :param d60: The numpy array of pixels's value for 60m resolution.
    :param deep: If true, 32 would be the number of resBlocks used in the model and 256
    would be the number of filters used to scan the image to create feature maps.
    :return: The predicted high resolution (10 m) image.
    """
    border = 12
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patchSize=192, border=border)
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
    if deep:
        model = s2model(input_shape, num_layers=32, feature_size=256)
        predict_file = MDL_PATH+'s2_034_lr_1e-04.hdf5' if run_60 else MDL_PATH+'s2_033_lr_1e-04.hdf5'
    else:
        model = s2model(input_shape, num_layers=6, feature_size=128)
        predict_file = MDL_PATH+'s2_030_lr_1e-05.hdf5' if run_60 else MDL_PATH+'s2_032_lr_1e-04.hdf5'
    print('Symbolic Model Created.')

    model.load_weights(predict_file)
    print("Predicting using file: {}".format(predict_file))
    prediction = model.predict(test, verbose=1)
    return prediction

