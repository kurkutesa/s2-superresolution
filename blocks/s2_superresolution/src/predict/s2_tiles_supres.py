import re
import sys
import os
from collections import defaultdict

from typing import List, Dict
from pathlib import Path
import glob

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import Affine as A
import pyproj as proj
from supres import DSen2_20, DSen2_60
from helper import get_logger, load_metadata, load_params, save_result, SENTINEL2_L1C

logger = get_logger(__name__)

# This code is adapted from this repository http://nicolas.brodu.net/code/superres and is distributed under the same
# license.


OUTPUT_DIR = '/tmp/output/'
INPUT_DIR = '/tmp/input/'
data_folder = '*/MTD*.xml'


def get_data(input_dir):
    """
    This method returns the raster data set of original image for all the available resolutions and the geojson file.
    :param input_dir: The directory to the original image.

    """
    input_metadata = load_metadata()
    for feature in input_metadata.features:
        path_to_input_img = feature["properties"][SENTINEL2_L1C]
        path_to_output_img = Path(path_to_input_img).stem + '_superresolution.tif'
        out_feature = feature.copy()
        out_feature["properties"]["custom.processing.superresolution"] = path_to_output_img
    for file in glob.iglob(os.path.join(input_dir, path_to_input_img, data_folder), recursive=True):
        DATA_PATH = file

    raster_data = rasterio.open(DATA_PATH)
    datasets = raster_data.subdatasets

    for dsdesc in datasets:
        if '10m' in dsdesc:
            d1 = rasterio.open(dsdesc)
        elif '20m' in dsdesc:
            d2 = rasterio.open(dsdesc)
        elif '60m' in dsdesc:
            d6 = rasterio.open(dsdesc)
        else:
            dunknown = rasterio.open(dsdesc)

    return d1, d2, d6, dunknown, out_feature, path_to_output_img


ds10, ds20, ds60, dsunknown, output_jsonfile, output_name = get_data(INPUT_DIR)


def get_max_min(x1, y1, x2, y2):
    """
    This method gets pixels' location for the region of interest on the 10m bands
    and returns the min/max in each direction and to nearby 60m pixel boundaries and the area
    associated to the region of interest.

    **Example**
    >>> get_max_min(0,0,400,400)
    (0, 0, 395, 395, 156816)

    """
    tmxmin = max(min(x1, x2, ds10.width - 1), 0)
    tmxmax = min(max(x1, x2, 0), ds10.width - 1)
    tmymin = max(min(y1, y2, ds10.height - 1), 0)
    tmymax = min(max(y1, y2, 0), ds10.height - 1)
    # enlarge to the nearest 60 pixel boundary for the super-resolution
    tmxmin = int(tmxmin / 6) * 6
    tmxmax = int((tmxmax + 1) / 6) * 6 - 1
    tmymin = int(tmymin / 6) * 6
    tmymax = int((tmymax + 1) / 6) * 6 - 1
    area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
    return tmxmin, tmymin, tmxmax, tmymax, area


def to_xy(lon, lat, data):
    """
    This method gets the longitude and the latitude of a given point and projects it
    into pixel location in the new coordinate system.

    :param lon: The longitude of a chosen point
    :param lat: The longitude of a chosen point
    :return: The pixel location in the coordinate system of the input image
    """
    # get the image's coordinate system.
    coor = data.transform
    a, b, xoff, d, e, yoff = [coor[x] for x in range(6)]

    # transform the lat and lon into x and y position which are defined in the world's coordinate system.
    crs_wgs = proj.Proj(init='epsg:4326')
    crs_bng = proj.Proj(init='epsg:32639')
    xp, yp = proj.transform(crs_wgs, crs_bng, lon, lat)
    xp -= xoff
    yp -= yoff

    # matrix inversion
    # get the x and y position in image's coordinate system.
    det_inv = 1. / (a * e - d * b)
    x = (e * xp - b * yp) * det_inv
    y = (-d * xp + a * yp) * det_inv
    return int(x), int(y)


params = load_params()  # type: dict


def area_of_interest():
    """
    This method returns the coordinates that define the desired area of interest.

    """
    if 'roi_x_y' in [*params]:
        roi_x1, roi_y1, roi_x2, roi_y2 = params.get('roi_x_y')['default']
        xmi, ymi, xma, yma, area = get_max_min(roi_x1, roi_y1, roi_x2, roi_y2)
    elif 'roi_lon_lat' in [*params]:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = params.get('roi_lon_lat')['default']
        x1, y1 = to_xy(roi_lon1, roi_lat1, ds10)
        x2, y2 = to_xy(roi_lon2, roi_lat2, ds10)
        xmi, ymi, xma, yma, area = get_max_min(x1, y1, x2, y2)
    else:
        xmi, ymi, xma, yma, area = (0, 0, ds10.width, ds10.height, ds10.width * ds10.height)

    return xmi, ymi, xma, yma, area


xmin, ymin, xmax, ymax, interest_area = area_of_interest()

ds10desc = ds10.crs.wkt
utm = ds10desc[ds10desc.find("UTM"):]

logger.info("Selected UTM Zone:")
logger.info(utm)
logger.info("Selected pixel region:")
logger.info('xmin = ' + str(xmin))
logger.info('ymin = ' + str(ymin))
logger.info('xmax = ' + str(xmax))
logger.info('ymax = ' + str(ymax))

if xmax < xmin or ymax < ymin:
    logger.info("Invalid region of interest / UTM Zone combination")
    sys.exit(0)


def validate_description(description):
    """
    This method rewrites the description of each band in the given data set.

    :param description: The actual description of a chosen band.

    **Example**
    >>> ds10.descriptions[0]
    'B4, central wavelength 665 nm'
    >>> validate_description(ds10.descriptions[0])
    'B4 (665 nm)'
    """
    m = re.match("(.*?), central wavelength (\d+) nm", description)
    if m:
        return m.group(1) + " (" + m.group(2) + " nm)"
    return description


def get_band_short_name(description):
    """
    This method returns only the name of the bands at a chosen resolution.

    :param description: This is the output of the validate_description method.

    **Example**
    >>> desc = validate_description(ds10.descriptions[0])
    >>> desc
    'B4 (665 nm)'
    >>> get_band_short_name(desc)
    'B4'
    """
    if ',' in description:
        return description[:description.find(',')]
    if ' ' in description:
        return description[:description.find(' ')]
    return description[:3]


def validate(data):
    """
    This method takes the short name of the bands for each separate resolution and returns
    three lists. The validated_bands and validated_indices contain the name of the bands and the indices
    related to them respectively. The validated_descriptions is a list of descriptions for each band
    obtained from the validate_description method.

    :param data: The raster file for a specific resolution.

    **Example**
    >>> validated_10m_bands, validated_10m_indices, dic_10m  = validate(ds10)
    >>> validated_10m_bands
    ['B4', 'B3', 'B2', 'B8']
    >>> validated_10m_indices
    [0, 1, 2, 3]
    >>> dic_10m
    defaultdict(<class 'str'>, {'B4': 'B4 (665 nm)', 'B3': 'B3 (560 nm)', 'B2': 'B2 (490 nm)', 'B8': 'B8 (842 nm)'})
    """
    select_bands = 'B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12'
    select_bands = [x for x in re.split(',', select_bands)]
    validated_bands = []
    validated_indices = []
    validated_descriptions = defaultdict(str)
    for i in range(0, data.count):
        desc = validate_description(data.descriptions[i])
        name = get_band_short_name(desc)
        if name in select_bands:
            select_bands.remove(name)
            validated_bands += [name]
            validated_indices += [i]
            validated_descriptions[name] = desc
    return validated_bands, validated_indices, validated_descriptions


logger.info("Selected 10m bands:")
validated_10m_bands, validated_10m_indices, dic_10m = validate(ds10)

logger.info("Selected 20m bands:")
validated_20m_bands, validated_20m_indices, dic_20m = validate(ds20)

logger.info("Selected 60m bands:")
validated_60m_bands, validated_60m_indices, dic_60m = validate(ds60)

validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}


def data_final(data, term, x_mi, y_mi, x_ma, y_ma, n):
    """
    This method takes the raster file at a specific resolution and uses the output of get_max_min
    to specify the area of interest. Then it returns an numpy array of values for all the pixels inside
    the area of interest.

    :param data: The raster file for a specific resolution.
    :param term: The validate indices of the bands obtained from the validate method.
    :return: The numpy array of pixels' value.
    """
    if term:
        print(term)
        d_final = np.rollaxis(
            data.read(window=Window(col_off=x_mi, row_off=y_mi, width=x_ma - x_mi + n, height=y_ma - y_mi + n)), 0, 3)[
                 :, :, term]
    return d_final


def run_model(d1, d2, d6):
    """
    This method takes the raster data at 10, 20, and 60 m resolutions and by applying fata_final method
    creates the input data for the the convolutional neural network. It returns 10 m resolution for all
    the bands in 20 and 60 m resolutions.

    :param d1: Raster data at 10m resolution.
    :param d2: Raster data at 20m resolution.
    :param d6: Raster data at 60m resolution.

    """
    data10 = data_final(d1, validated_10m_indices, xmin, ymin, xmax, ymax, 1)
    data20 = data_final(d2, validated_20m_indices, xmin // 2, ymin // 2, xmax // 2, ymax // 2, 1 // 2)
    data60 = data_final(d6, validated_60m_indices, xmin // 6, ymin // 6, xmax // 6, ymax // 6, 1 // 6)

    if validated_60m_bands and validated_20m_bands and validated_10m_bands:
        logger.info("Super-resolving the 60m data into 10m bands")
        sr60 = DSen2_60(data10, data20, data60, deep=False)
        logger.info("Super-resolving the 20m data into 10m bands")
        sr20 = DSen2_20(data10, data20, deep=False)
        sr_final = np.concatenate((sr20, sr60), axis=2)
        validated_sr_final_bands = validated_20m_bands + validated_60m_bands
    else:
        sr_final = None
        validated_sr_final_bands = None
        logger.info("No super-resolution performed, exiting")
        sys.exit(0)
    return sr_final, validated_sr_final_bands, data10.shape


sr, validated_sr_bands, shape_10m = run_model(ds10, ds20, ds60)


if params['copy_original_bands']['default'] == 'yes':
    out_dims = shape_10m[2] + sr.shape[2]
else:
    out_dims = sr.shape[2]

logger.info("Writing")
logger.info(" the super-resolved bands in")


def update(data):
    """
    This method creates the proper georeferencing for the output image.
    :param data: The raster file for 10m resolution.

    """
    p = data.profile
    new_transform = p['transform'] * A.translation(xmin, ymin)
    p.update(dtype=rasterio.float32)
    p.update(driver='GTiff')
    p.update(width=shape_10m[1])
    p.update(height=shape_10m[0])
    p.update(count=out_dims)
    p.update(transform=new_transform)
    return p


profile = update(ds10)
filename = os.path.join(OUTPUT_DIR, output_name)
save_result(sr, validated_sr_bands, validated_descriptions_all, profile, output_jsonfile, OUTPUT_DIR, filename)


