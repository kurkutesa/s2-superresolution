"""
This module creates small patches from image to make the training process faster.
"""
from random import randrange
import os
import glob
from typing import Tuple
import json
from math import ceil

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import skimage.measure

from helper import get_logger

LOGGER = get_logger(__name__)

# This code is adapted from this repository
# https://github.com/lanha/DSen2 and is distributed under the same
# license.


def interp_patches(image_20, image_10_shape) -> np.ndarray:
    """
    This method resize the image to match a certain size.
    :param image_20: Input image of 20m resolution.
    :param image_10_shape: The pixel size of the image at 10m resolutions.
    :return: A resized image.
    """
    data20_interp = np.zeros((image_20.shape[0:2] + image_10_shape[2:4])).astype(
        np.float32
    )
    for k_i in range(image_20.shape[0]):
        for w_i in range(image_20.shape[1]):
            data20_interp[k_i, w_i] = (
                resize(image_20[k_i, w_i] / 30000, image_10_shape[2:4], mode="reflect")
                * 30000
            )  # bilinear
    return data20_interp


# pylint: disable-msg=too-many-locals
def get_test_patches(dset_10, dset_20, patchsize=128, border=4, interp=True) -> Tuple:
    """
    This method create the test patches of for testing the model to create 10m resolution
    for all the spectral bands at 20m.
    """
    patch_size_hr = (patchsize, patchsize)
    patch_size_lr = [p // 2 for p in patch_size_hr]
    border_hr = border
    border_lr = border_hr // 2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10,
        ((border_hr, border_hr), (border_hr, border_hr), (0, 0)),
        mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_lr, border_lr), (border_lr, border_lr), (0, 0)),
        mode="symmetric",
    )

    bands10 = dset_10.shape[2]
    bands20 = dset_20.shape[2]
    patches_alongi = (dset_20.shape[0] - 2 * border_lr) // (
        patch_size_lr[0] - 2 * border_lr
    )
    patches_alongj = (dset_20.shape[1] - 2 * border_lr) // (
        patch_size_lr[1] - 2 * border_lr
    )

    nr_patches = (patches_alongi + 1) * (patches_alongj + 1)

    # pylint: disable=unused-variable
    label_20 = np.zeros((nr_patches, bands20) + patch_size_hr).astype(np.float32)
    image_20 = np.zeros((nr_patches, bands20) + tuple(patch_size_lr)).astype(np.float32)
    image_10 = np.zeros((nr_patches, bands10) + patch_size_hr).astype(np.float32)

    range_i = np.arange(
        0, (dset_20.shape[0] - 2 * border_lr) // (patch_size_lr[0] - 2 * border_lr)
    ) * (patch_size_lr[0] - 2 * border_lr)
    range_j = np.arange(
        0, (dset_20.shape[1] - 2 * border_lr) // (patch_size_lr[1] - 2 * border_lr)
    ) * (patch_size_lr[1] - 2 * border_lr)

    if (
        not np.mod(dset_20.shape[0] - 2 * border_lr, patch_size_lr[0] - 2 * border_lr)
        == 0
    ):
        range_i = np.append(range_i, (dset_20.shape[0] - patch_size_lr[0]))
    if (
        not np.mod(dset_20.shape[1] - 2 * border_lr, patch_size_lr[1] - 2 * border_lr)
        == 0
    ):
        range_j = np.append(range_j, (dset_20.shape[1] - patch_size_lr[1]))

    p_count = 0
    for i_r in range_i.astype(int):
        for j_r in range_j.astype(int):
            upper_left_i = i_r
            upper_left_j = j_r
            crop_point_lr = [
                upper_left_i,
                upper_left_j,
                upper_left_i + patch_size_lr[0],
                upper_left_j + patch_size_lr[1],
            ]
            crop_point_hr = [p * 2 for p in crop_point_lr]
            image_20[p_count] = np.rollaxis(
                dset_20[
                    crop_point_lr[0] : crop_point_lr[2],
                    crop_point_lr[1] : crop_point_lr[3],
                ],
                2,
            )
            image_10[p_count] = np.rollaxis(
                dset_10[
                    crop_point_hr[0] : crop_point_hr[2],
                    crop_point_hr[1] : crop_point_hr[3],
                ],
                2,
            )
            p_count += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
    else:
        data20_interp = image_20
    return image_10, data20_interp


# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-arguments
def get_test_patches60(
    dset_10, dset_20, dset_60, patchsize=128, border=8, interp=True
) -> Tuple:
    """
    This method create the test patches of for testing the model to create 10m resolution
    for all the spectral bands at 20m and 60m.
    """
    patch_size_10 = (patchsize, patchsize)
    patch_size_20 = [p // 2 for p in patch_size_10]
    patch_size_60 = [p // 6 for p in patch_size_10]
    border_10 = border
    border_20 = border_10 // 2
    border_60 = border_10 // 6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10,
        ((border_10, border_10), (border_10, border_10), (0, 0)),
        mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_20, border_20), (border_20, border_20), (0, 0)),
        mode="symmetric",
    )
    dset_60 = np.pad(
        dset_60,
        ((border_60, border_60), (border_60, border_60), (0, 0)),
        mode="symmetric",
    )

    bands10 = dset_10.shape[2]
    bands20 = dset_20.shape[2]
    bands60 = dset_60.shape[2]
    patches_alongi = (dset_60.shape[0] - 2 * border_60) // (
        patch_size_60[0] - 2 * border_60
    )
    patches_alongj = (dset_60.shape[1] - 2 * border_60) // (
        patch_size_60[1] - 2 * border_60
    )

    nr_patches = (patches_alongi + 1) * (patches_alongj + 1)

    image_10 = np.zeros((nr_patches, bands10) + patch_size_10).astype(np.float32)
    image_20 = np.zeros((nr_patches, bands20) + tuple(patch_size_20)).astype(np.float32)
    image_60 = np.zeros((nr_patches, bands60) + tuple(patch_size_60)).astype(np.float32)

    range_i = np.arange(
        0, (dset_60.shape[0] - 2 * border_60) // (patch_size_60[0] - 2 * border_60)
    ) * (patch_size_60[0] - 2 * border_60)
    range_j = np.arange(
        0, (dset_60.shape[1] - 2 * border_60) // (patch_size_60[1] - 2 * border_60)
    ) * (patch_size_60[1] - 2 * border_60)

    if (
        not np.mod(dset_60.shape[0] - 2 * border_60, patch_size_60[0] - 2 * border_60)
        == 0
    ):
        range_i = np.append(range_i, (dset_60.shape[0] - patch_size_60[0]))
    if (
        not np.mod(dset_60.shape[1] - 2 * border_60, patch_size_60[1] - 2 * border_60)
        == 0
    ):
        range_j = np.append(range_j, (dset_60.shape[1] - patch_size_60[1]))

    p_count = 0
    for i_r in range_i.astype(int):
        for j_r in range_j.astype(int):
            upper_left_i = i_r
            upper_left_j = j_r
            crop_point_60 = [
                upper_left_i,
                upper_left_j,
                upper_left_i + patch_size_60[0],
                upper_left_j + patch_size_60[1],
            ]
            crop_point_10 = [p * 6 for p in crop_point_60]
            crop_point_20 = [p * 3 for p in crop_point_60]
            image_10[p_count] = np.rollaxis(
                dset_10[
                    crop_point_10[0] : crop_point_10[2],
                    crop_point_10[1] : crop_point_10[3],
                ],
                2,
            )
            image_20[p_count] = np.rollaxis(
                dset_20[
                    crop_point_20[0] : crop_point_20[2],
                    crop_point_20[1] : crop_point_20[3],
                ],
                2,
            )
            image_60[p_count] = np.rollaxis(
                dset_60[
                    crop_point_60[0] : crop_point_60[2],
                    crop_point_60[1] : crop_point_60[3],
                ],
                2,
            )
            p_count += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
        data60_interp = interp_patches(image_60, image_10_shape)

    else:
        data20_interp = image_20
        data60_interp = image_60

    return image_10, data20_interp, data60_interp


# pylint: disable-msg=too-many-arguments
def save_test_patches(dset_10, dset_20, file, patchsize=128, border=4, interp=True):
    """
    This methods save all testing patches to the defined directory. So that they will
    be used as input for the CNN for create the 10m resolution for all the spectral
    bands at 20m resolution.
    """
    image_10, data20_interp = get_test_patches(
        dset_10, dset_20, patchsize=patchsize, border=border, interp=interp
    )

    LOGGER.info("Saving to file %s", file)

    np.save(file + "data10", image_10)
    np.save(file + "data20", data20_interp)
    LOGGER.info("Done!")


# pylint: disable-msg=too-many-arguments
def save_test_patches60(
    dset_10, dset_20, dset_60, file, patchsize=192, border=12, interp=True
):
    """
    This methods save all testing patches to the defined directory. So that they will
    be used as input for the CNN for create the 10m resolution for all the spectral
    bands at 20m and 60m resolution.
    """
    image_10, data20_interp, data60_interp = get_test_patches60(
        dset_10, dset_20, dset_60, patchsize=patchsize, border=border, interp=interp
    )
    LOGGER.info("Saving to file %s", file)

    np.save(file + "data10", image_10)
    np.save(file + "data20", data20_interp)
    np.save(file + "data60", data60_interp)
    LOGGER.info("Done!")


# pylint: disable-msg=too-many-locals
def save_random_patches(dset_20gt, dset_10, dset_20, file, nr_crop=8000):
    """
    This method creates random patches based on the input image for training the model.
    """
    patch_size_hr = (32, 32)
    patch_size_lr = (16, 16)

    bands10 = dset_10.shape[2]
    bands20 = dset_20.shape[2]
    label_20 = np.zeros((nr_crop, bands20) + patch_size_hr).astype(np.float32)
    image_20 = np.zeros((nr_crop, bands20) + patch_size_lr).astype(np.float32)
    image_10 = np.zeros((nr_crop, bands10) + patch_size_hr).astype(np.float32)

    i = 0
    for crop in range(0, nr_crop):  # pylint: disable=unused-variable
        # while True:
        upper_left_x = randrange(0, dset_20.shape[0] - patch_size_lr[0])
        upper_left_y = randrange(0, dset_20.shape[1] - patch_size_lr[1])
        crop_point_lr = [
            upper_left_x,
            upper_left_y,
            upper_left_x + patch_size_lr[0],
            upper_left_y + patch_size_lr[1],
        ]
        crop_point_hr = [p * 2 for p in crop_point_lr]
        label_20[i] = np.rollaxis(
            dset_20gt[
                crop_point_hr[0] : crop_point_hr[2], crop_point_hr[1] : crop_point_hr[3]
            ],
            2,
        )
        image_20[i] = np.rollaxis(
            dset_20[
                crop_point_lr[0] : crop_point_lr[2], crop_point_lr[1] : crop_point_lr[3]
            ],
            2,
        )
        image_10[i] = np.rollaxis(
            dset_10[
                crop_point_hr[0] : crop_point_hr[2], crop_point_hr[1] : crop_point_hr[3]
            ],
            2,
        )
        i += 1
    np.save(file + "data10", image_10)
    image_10_shape = image_10.shape
    del image_10
    np.save(file + "data20_gt", label_20)
    del label_20

    data20_interp = interp_patches(image_20, image_10_shape)
    np.save(file + "data20", data20_interp)

    LOGGER.info("Done!")


# pylint: disable-msg=too-many-locals
# pylint: disable-msg=too-many-arguments
def save_random_patches60(dset_60gt, dset_10, dset_20, dset_60, file, nr_crop=500):
    """
    This method creates random patches based on the input image for training the model.
    """
    patch_size_10 = (96, 96)
    patch_size_20 = (48, 48)
    patch_size_60 = (16, 16)

    bands10 = dset_10.shape[2]
    bands20 = dset_20.shape[2]
    bands60 = dset_60.shape[2]
    label_60 = np.zeros((nr_crop, bands60) + patch_size_10).astype(np.float32)
    image_10 = np.zeros((nr_crop, bands10) + patch_size_10).astype(np.float32)
    image_20 = np.zeros((nr_crop, bands20) + patch_size_20).astype(np.float32)
    image_60 = np.zeros((nr_crop, bands60) + patch_size_60).astype(np.float32)

    print(label_60.shape)
    print(image_10.shape)
    print(image_20.shape)
    print(image_60.shape)

    i = 0
    for crop in range(0, nr_crop):  # pylint: disable=unused-variable
        # while True:
        upper_left_x = randrange(0, dset_60.shape[0] - patch_size_60[0])
        upper_left_y = randrange(0, dset_60.shape[1] - patch_size_60[1])
        crop_point_lr = [
            upper_left_x,
            upper_left_y,
            upper_left_x + patch_size_60[0],
            upper_left_y + patch_size_60[1],
        ]
        crop_point_hr20 = [p * 3 for p in crop_point_lr]
        crop_point_hr60 = [p * 6 for p in crop_point_lr]

        label_60[i] = np.rollaxis(
            dset_60gt[
                crop_point_hr60[0] : crop_point_hr60[2],
                crop_point_hr60[1] : crop_point_hr60[3],
            ],
            2,
        )
        image_10[i] = np.rollaxis(
            dset_10[
                crop_point_hr60[0] : crop_point_hr60[2],
                crop_point_hr60[1] : crop_point_hr60[3],
            ],
            2,
        )
        image_20[i] = np.rollaxis(
            dset_20[
                crop_point_hr20[0] : crop_point_hr20[2],
                crop_point_hr20[1] : crop_point_hr20[3],
            ],
            2,
        )
        image_60[i] = np.rollaxis(
            dset_60[
                crop_point_lr[0] : crop_point_lr[2], crop_point_lr[1] : crop_point_lr[3]
            ],
            2,
        )
        i += 1
    np.save(file + "data10", image_10)
    image_10_shape = image_10.shape
    del image_10
    np.save(file + "data60_gt", label_60)
    del label_60

    data20_interp = interp_patches(image_20, image_10_shape)
    np.save(file + "data20", data20_interp)
    del data20_interp

    data60_interp = interp_patches(image_60, image_10_shape)
    np.save(file + "data60", data60_interp)

    LOGGER.info("Done!")


def splittrainval(train_path, train, label) -> Tuple:
    """
    This method is used for splitting the input image into training, testing and validation
    sets.
    """
    try:
        val_ind = np.load(train_path + "val_index.npy")
    except IOError:
        LOGGER.debug(
            "Please define the validation split indices,"
            " usually located in .../data/test/. To generate this file use"
            " createRandom.py"
        )
    val_tr = [
        p[val_ind] for p in train
    ]  # pylint: disable-msg=invalid-unary-operand-type
    train = [
        p[~val_ind] for p in train
    ]  # pylint: disable-msg=invalid-unary-operand-type
    val_lb = label[val_ind]  # pylint: disable-msg=invalid-unary-operand-type
    label = label[~val_ind]  # pylint: disable-msg=invalid-unary-operand-type
    LOGGER.info("Loaded %s patches for training.", val_ind.shape[0])
    return train, label, val_tr, val_lb


def opendatafiles(path, run_60, scale):
    """
    This method opens the relevant path that the training data has been saved. Then
    the splittrainval method will be called to make the splitting.
    """
    if run_60:
        train_path = path + "train60/"
    else:
        train_path = path + "train/"
    # Initialize in able to concatenate
    data20_gt = data60_gt = data10 = data20 = data60 = None
    # train = label = None
    # Create list from path
    file_list = [os.path.basename(x) for x in sorted(glob.glob(train_path + "*SAFE"))]
    for dset in file_list:
        data10_new = np.load(train_path + dset + "/data10.npy")
        data20_new = np.load(train_path + dset + "/data20.npy")
        data10 = (
            np.concatenate((data10, data10_new)) if data10 is not None else data10_new
        )
        data20 = (
            np.concatenate((data20, data20_new)) if data20 is not None else data20_new
        )
        if run_60:
            data60_gt_new = np.load(train_path + dset + "/data60_gt.npy")
            data60_new = np.load(train_path + dset + "/data60.npy")
            data60_gt = (
                np.concatenate((data60_gt, data60_gt_new))
                if data60_gt is not None
                else data60_gt_new
            )
            data60 = (
                np.concatenate((data60, data60_new))
                if data60 is not None
                else data60_new
            )
        else:
            data20_gt_new = np.load(train_path + dset + "/data20_gt.npy")
            data20_gt = (
                np.concatenate((data20_gt, data20_gt_new))
                if data20_gt is not None
                else data20_gt_new
            )

    if scale:
        data10 /= scale
        data20 /= scale
        if run_60:
            data60 /= scale
            data60_gt /= scale
        else:
            data20_gt /= scale

    if run_60:
        return splittrainval(train_path, [data10, data20, data60], data60_gt)

    return splittrainval(train_path, [data10, data20], data20_gt)


def opendatafilestest(path, run_60, scale, true_scale=False) -> Tuple:
    """
    This method return the image that will be used for testing the model.
    """
    if not scale:
        scale = 1

    data10 = np.load(path + "/data10.npy")
    data20 = np.load(path + "/data20.npy")
    data10 /= scale
    data20 /= scale
    if run_60:
        data60 = np.load(path + "/data60.npy")
        data60 /= scale
        train = [data10, data20, data60]
    else:
        train = [data10, data20]

    with open(path + "/roi.json") as data_file:
        data = json.load(data_file)

    image_size = [(data[2] - data[0]), (data[3] - data[1])]

    LOGGER.info("The image size is: %s}", image_size)
    LOGGER.info("The SCALE is: %s", scale)
    LOGGER.info("The true_scale is: %s", true_scale)
    return train, image_size


def downpixelaggr(img, scale=2):
    """
    This method use a Gaussian filter for blurring the original image.
    """

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_blur = np.zeros(img.shape)
    # Filter the image with a Gaussian filter
    for i in range(0, img.shape[2]):
        img_blur[:, :, i] = gaussian_filter(img[:, :, i], 1 / scale)
    # New image dims
    new_dims = tuple(s // scale for s in img.shape)
    img_lr = np.zeros(new_dims[0:2] + (img.shape[-1],))
    # Iterate through all the image channels with avg pooling (pixel aggregation)
    for i in range(0, img.shape[2]):
        img_lr[:, :, i] = skimage.measure.block_reduce(
            img_blur[:, :, i], (scale, scale), np.mean
        )

    return np.squeeze(img_lr)


def recompose_images(a_re, border, size=None):
    """
    This method attached all the patches from prediction to create the final output image.
    """
    if a_re.shape[0] == 1:
        images = a_re[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a_re.shape[2] - border * 2

        x_tiles = int(ceil(size[1] / float(patch_size)))
        y_tiles = int(ceil(size[0] / float(patch_size)))

        # Initialize image
        images = np.zeros((a_re.shape[1], size[0], size[1])).astype(np.float32)

        LOGGER.info(images.shape)
        current_patch = 0
        for y_t in range(0, y_tiles):
            ypoint = y_t * patch_size
            if ypoint > size[0] - patch_size:
                ypoint = size[0] - patch_size
            for x_t in range(0, x_tiles):
                xpoint = x_t * patch_size
                if xpoint > size[1] - patch_size:
                    xpoint = size[1] - patch_size
                images[
                    :, ypoint : ypoint + patch_size, xpoint : xpoint + patch_size
                ] = a_re[
                    current_patch,
                    :,
                    border : a_re.shape[2] - border,
                    border : a_re.shape[3] - border,
                ]
                current_patch += 1

    return images.transpose((1, 2, 0))
