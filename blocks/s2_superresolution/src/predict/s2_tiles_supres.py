"""
This module is the main script for creating super-resolution spectral bands from Sentinel-2 images.
"""
import re
import sys
import os
from collections import defaultdict

from typing import List, Tuple
from pathlib import Path
import glob

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import Affine as A
import pyproj as proj
from supres import dsen2_20, dsen2_60
from helper import get_logger, load_metadata, load_params, \
        save_result, SENTINEL2_L1C, ensure_data_directories_exist

LOGGER = get_logger(__name__)

# This code is adapted from this repository
# http://nicolas.brodu.net/code/superres and is distributed under the same
# license.


class Superresolution:
    """
    This class implements a CNN model to obtain a high resolution (10m)
    bands for 20m and 60m resolution.
    """

    def __init__(self, output_dir: str = '/tmp/output/',
                 input_dir: str = '/tmp/input/',
                 data_folder: str = '*/MTD*.xml'):
        """
        :param output_dir: The directory for the output image.
        :param input_dir: The directory of the original image.
        :param data_folder: The original image file.
        """
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.data_folder = data_folder

    def get_data(self) -> Tuple:
        """
        This method returns the raster data set of original image for
        all the available resolutions and the geojson file.
        :param input_loc: The directory to the original image.

        """
        path_to_input_img = None
        data_path = None
        input_metadata = load_metadata()
        for feature in input_metadata.features:
            path_to_input_img = feature["properties"][SENTINEL2_L1C]
            path_to_output_img = Path(path_to_input_img).stem + \
                '_superresolution.tif'
            out_feature = feature.copy()
            out_feature["properties"]["custom.processing.superresolution"] =\
                path_to_output_img
        for file in glob.iglob(os.path.join(self.input_dir, str(path_to_input_img),
                                            self.data_folder), recursive=True):
            data_path = file

        raster_data = rasterio.open(data_path)
        datasets = raster_data.subdatasets

        for dsdesc in datasets:
            if '10m' in dsdesc:
                d_1 = rasterio.open(dsdesc)
            elif '20m' in dsdesc:
                d_2 = rasterio.open(dsdesc)
            elif '60m' in dsdesc:
                d_6 = rasterio.open(dsdesc)

        return d_1, d_2, d_6, out_feature, path_to_output_img

    @staticmethod
    # pylint: disable-msg=too-many-locals
    def get_max_min(x_1: int, y_1: int, x_2: int, y_2: int, data) -> Tuple:
        # pylint: disable = R0914
        """
        This method gets pixels' location for the region of interest on the 10m bands
        and returns the min/max in each direction and to nearby 60m pixel boundaries and the area
        associated to the region of interest.
        **Example**
        >>> get_max_min(0,0,400,400)
        (0, 0, 395, 395, 156816)

        """
        tmxmin = max(min(x_1, x_2, data.width - 1), 0)
        tmxmax = min(max(x_1, x_2, 0), data.width - 1)
        tmymin = max(min(y_1, y_2, data.height - 1), 0)
        tmymax = min(max(y_1, y_2, 0), data.height - 1)
        # enlarge to the nearest 60 pixel boundary for the super-resolution
        tmxmin = int(tmxmin / 6) * 6
        tmxmax = int((tmxmax + 1) / 6) * 6 - 1
        tmymin = int(tmymin / 6) * 6
        tmymax = int((tmymax + 1) / 6) * 6 - 1
        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        return tmxmin, tmymin, tmxmax, tmymax, area

    # pylint: disable-msg=too-many-locals
    def to_xy(self, lon: float, lat: float, data) -> Tuple:
        """
        This method gets the longitude and the latitude of a given point and projects it
        into pixel location in the new coordinate system.
        :param lon: The longitude of a chosen point
        :param lat: The longitude of a chosen point
        :return: The pixel location in the coordinate system of the input image
        """
        # get the image's coordinate system.
        coor = data.transform
        a_t, b_t, xoff, d_t, e_t, yoff = [coor[x] for x in range(6)]

        # transform the lat and lon into x and y position which are defined in
        # the world's coordinate system.
        local_crs = self.get_utm(data)
        crs_wgs = proj.Proj(init='epsg:4326')  # WGS 84 geographic coordinate system
        crs_bng = proj.Proj(init=local_crs)  # use a locally appropriate projected CRS
        x_p, y_p = proj.transform(crs_wgs, crs_bng, lon, lat)
        x_p -= xoff
        y_p -= yoff

        # matrix inversion
        # get the x and y position in image's coordinate system.
        det_inv = 1. / (a_t * e_t - d_t * b_t)
        x_n = (e_t * x_p - b_t * y_p) * det_inv
        y_n = (-d_t * x_p + a_t * y_p) * det_inv
        return int(x_n), int(y_n)

    @staticmethod
    def get_utm(data) -> str:
        """
        This method returns the utm of the input image.
        :param data: The raster file for a specific resolution.
        :return: UTM of the selected raster file.
        """
        data_crs = data.crs.to_dict()
        utm = data_crs['init']
        return utm

    # pylint: disable-msg=too-many-locals
    def area_of_interest(self, data):
        """
        This method returns the coordinates that define the desired area of interest.
        """
        params = load_params()
        if 'roi_x_y' in [*params]:
            roi_x1, roi_y1, roi_x2, roi_y2 = params.get('roi_x_y')
            xmi, ymi, xma, yma, area = self.get_max_min(roi_x1, roi_y1, roi_x2, roi_y2, data)
        elif 'roi_lon_lat' in [*params]:
            roi_lon1, roi_lat1, roi_lon2, roi_lat2 = params.get('roi_lon_lat')
            x_1, y_1 = self.to_xy(roi_lon1, roi_lat1, data)
            x_2, y_2 = self.to_xy(roi_lon2, roi_lat2, data)
            xmi, ymi, xma, yma, area = self.get_max_min(x_1, y_1, x_2, y_2, data)
        else:
            xmi, ymi, xma, yma, area = (0, 0, data.width, data.height, data.width * data.height)

        return xmi, ymi, xma, yma, area

    @staticmethod
    def validate_description(description: str) -> str:
        """
        This method rewrites the description of each band in the given data set.
        :param description: The actual description of a chosen band.

        **Example**
        >>> ds10.descriptions[0]
        'B4, central wavelength 665 nm'
        >>> validate_description(ds10.descriptions[0])
        'B4 (665 nm)'
        """
        m_re = re.match(r"(.*?), central wavelength (\d+) nm", description)
        if m_re:
            return m_re.group(1) + " (" + m_re.group(2) + " nm)"
        return description

    @staticmethod
    def get_band_short_name(description: str) -> str:
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

    def validate(self, data) -> Tuple:
        """
        This method takes the short name of the bands for each
        separate resolution and returns three lists. The validated_
        bands and validated_indices contain the name of the bands and
        the indices related to them respectively.
        The validated_descriptions is a list of descriptions for each band
        obtained from the validate_description method.
        :param data: The raster file for a specific resolution.
        **Example**
        >>> validated_10m_bands, validated_10m_indices, \
        >>> dic_10m = validate(ds10)
        >>> validated_10m_bands
        ['B4', 'B3', 'B2', 'B8']
        >>> validated_10m_indices
        [0, 1, 2, 3]
        >>> dic_10m
        defaultdict(<class 'str'>, {'B4': 'B4 (665 nm)',
         'B3': 'B3 (560 nm)', 'B2': 'B2 (490 nm)', 'B8': 'B8 (842 nm)'})
        """
        input_select_bands = 'B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12'  # type: str
        select_bands = [x for x in re.split(',', input_select_bands)]  # type: List[str]
        validated_bands = []  # type: list
        validated_indices = []  # type: list
        validated_descriptions = defaultdict(str)  # type: defaultdict
        for i in range(0, data.count):
            desc = self.validate_description(data.descriptions[i])
            name = self.get_band_short_name(desc)
            if name in select_bands:
                select_bands.remove(name)
                validated_bands += [name]
                validated_indices += [i]
                validated_descriptions[name] = desc
        return validated_bands, validated_indices, \
               validated_descriptions

    @staticmethod
    # pylint: disable-msg=too-many-arguments
    def data_final(data, term: List, x_mi: int, y_mi: int,
                   x_ma: int, y_ma: int, n_res) -> np.ndarray:
        """
        This method takes the raster file at a specific
        resolution and uses the output of get_max_min
        to specify the area of interest.
        Then it returns an numpy array of values
        for all the pixels inside the area of interest.
        :param data: The raster file for a specific resolution.
        :param term: The validate indices of the
        bands obtained from the validate method.
        :return: The numpy array of pixels' value.
        """
        if term:
            print(term)
            d_final = np.rollaxis(
                data.read(window=Window(col_off=x_mi, row_off=y_mi,
                                        width=x_ma - x_mi + n_res, height=y_ma - y_mi + n_res))
                , 0, 3)[:, :, term]
        return d_final

    # pylint: disable-msg=too-many-locals
    def run_model(self, d_1, d_2, d_6) -> Tuple:
        """
        This method takes the raster data at 10,
        20, and 60 m resolutions and by applying
        data_final method creates the input data
        for the the convolutional neural network.
        It returns 10 m resolution for all
        the bands in 20 and 60 m resolutions.
        :param d_1: Raster data at 10m resolution.
        :param d_2: Raster data at 20m resolution.
        :param d_6: Raster data at 60m resolution.

        """
        xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(d_1)
        LOGGER.info("Selected pixel region:")
        LOGGER.info('xmin = %s', xmin)
        LOGGER.info('ymin = %s', ymin)
        LOGGER.info('xmax = %s', xmax)
        LOGGER.info('ymax = %s', ymax)
        LOGGER.info('The area of selected region = %s', interest_area)
        if xmax < xmin or ymax < ymin:
            LOGGER.info("Invalid region of interest / UTM Zone combination")
            sys.exit(0)

        LOGGER.info("Selected 10m bands:")
        validated_10m_bands, validated_10m_indices, dic_10m = self.validate(d_1)

        LOGGER.info("Selected 20m bands:")
        validated_20m_bands, validated_20m_indices, dic_20m = self.validate(d_2)

        LOGGER.info("Selected 60m bands:")
        validated_60m_bands, validated_60m_indices, dic_60m = self.validate(d_6)

        validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}

        data10 = self.data_final(d_1, validated_10m_indices,
                                 xmin, ymin, xmax, ymax, 1)
        data20 = self.data_final(d_2, validated_20m_indices,
                                 xmin // 2, ymin // 2, xmax // 2, ymax // 2, 1 // 2)
        data60 = self.data_final(d_6, validated_60m_indices,
                                 xmin // 6, ymin // 6, xmax // 6, ymax // 6, 1 // 6)

        if validated_60m_bands and validated_20m_bands and validated_10m_bands:
            LOGGER.info("Super-resolving the 60m data into 10m bands")
            sr60 = dsen2_60(data10, data20, data60, deep=False)
            LOGGER.info("Super-resolving the 20m data into 10m bands")
            sr20 = dsen2_20(data10, data20, deep=False)
            sr_final = np.concatenate((sr20, sr60), axis=2)
            validated_sr_final_bands = validated_20m_bands + validated_60m_bands
        else:
            LOGGER.info("No super-resolution performed, exiting")
            sys.exit(0)

        p_r = self.update(d_1, data10.shape, sr_final, xmin, ymin)
        return sr_final, validated_sr_final_bands, validated_descriptions_all, p_r

    @staticmethod
    def update(data, size_10m: Tuple, model_output: np.ndarray, xmi: int, ymi: int):
        """
        This method creates the proper georeferencing for the output image.
        :param data: The raster file for 10m resolution.

        """
        # Here based on the params.json file, the output image dimension will be calculated.
        params = load_params()  # type: dict
        if params['copy_original_bands'] == 'true':
            out_dims = size_10m[2] + model_output.shape[2]
        else:
            out_dims = model_output.shape[2]

        p_r = data.profile
        new_transform = p_r['transform'] * A.translation(xmi, ymi)
        p_r.update(dtype=rasterio.float32)
        p_r.update(driver='GTiff')
        p_r.update(width=size_10m[1])
        p_r.update(height=size_10m[0])
        p_r.update(count=out_dims)
        p_r.update(transform=new_transform)
        return p_r

    @staticmethod
    def run():
        """
        This method is the main entry point for this processing block
        """
        ensure_data_directories_exist()
        srr = Superresolution()
        ds10, ds20, ds60, output_jsonfile, output_name = srr.get_data()
        s_r, validated_sr_bands, validated_desc_all, profile = \
            srr.run_model(ds10, ds20, ds60)
        filename = os.path.join(srr.output_dir, output_name)
        LOGGER.info("Now writing the super-resolved bands")
        save_result(s_r, validated_sr_bands, validated_desc_all,
                    profile, output_jsonfile, srr.output_dir, filename)
