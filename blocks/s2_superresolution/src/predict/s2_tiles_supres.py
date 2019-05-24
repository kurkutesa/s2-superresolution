import argparse
import re
import sys
import os
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import Affine as A
import pyproj as proj
from supres import DSen2_20, DSen2_60
from helper import get_logger

logger = get_logger(__name__)

# This code is adapted from this repository http://nicolas.brodu.net/code/superres and is distributed under the same
# license.

parser = argparse.ArgumentParser(description="Perform super-resolution on Sentinel-2 with DSen2. Code based on superres"
                                             " by Nicolas Brodu.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("data_file",
                    help="An input sentinel-2 data file. This is the S2A[...].xml "
                         "file in a SAFE directory extracted from that ZIP.")
parser.add_argument("output_file", nargs="?",
                    help="A target data file. See also the --save_prefix option, and the --output_file_format option "
                         "(default is GTiff).")
parser.add_argument("--roi_lon_lat", default="",
                    help="Sets the region of interest to extract, WGS84, decimal notation. Use this syntax: lon_1,"
                         "lat_1,lon_2,lat_2. The order of points 1 and 2 does not matter: the region of interest "
                         "extends to the min/max in each direction. "
                         "Example: --roi_lon_lat=-1.12132,44.72408,-0.90350,44.58646")
parser.add_argument("--roi_x_y", default="",
                    help="Sets the region of interest to extract as pixels locations on the 10m bands. Use this "
                         "syntax: x_1,y_1,x_2,y_2. The order of points 1 and 2 does not matter: the region of interest "
                         "extends to the min/max in each direction and to nearby 60m pixel boundaries.")
parser.add_argument("--list_bands", action="store_true",
                    help="List bands in the input file subdata set matching the selected UTM zone, and exit.")
parser.add_argument("--run_60", action="store_true",
                    help="Select which bands to process and include in the output file. If this flag is set it will "
                         "super-resolve the 20m and 60m bands (B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12). If it is not "
                         "set it will only super-resolve the 20m bands (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12). Band B10 "
                         "is to noisy and is not super-resolved.")
parser.add_argument("--list_UTM", action="store_true",
                    help="List all UTM zones present in the input file, together with their coverage of the ROI in "
                         "10m x 10m pixels.")
parser.add_argument("--select_UTM", default="",
                    help="Select a UTM zone. The default is to select the zone with the largest coverage of the ROI.")
parser.add_argument("--list_output_file_formats", action="store_true",
                    help="If specified, list all supported raster output file formats declared by GDAL and exit. Some "
                         "of these formats may be inappropriate for storing Sentinel-2 multispectral data.")
parser.add_argument("--output_file_format", default="GTiff",
                    help="Speficies the name of a GDAL driver that supports file creation, like ENVI or GTiff. If no "
                         "such driver exists, or if the format is \"npz\", then save all bands instead as a compressed "
                         "python/numpy file")
parser.add_argument("--copy_original_bands", action="store_true",
                    help="The default is not to copy the original selected 10m bands into the output file in addition "
                         "to the super-resolved bands. If this flag is used, the output file may be used as a 10m "
                         "version of the original Sentinel-2 file.")
parser.add_argument("--save_prefix", default="",
                    help="If set, specifies the name of a prefix for all output files. Use a trailing / to save into a "
                         "directory. The default of no prefix will save into the current directory. "
                         "Example: --save_prefix result/")

args = parser.parse_args()
globals().update(args.__dict__)

OUTPUT_DIR = '/tmp/output/'

if run_60:
    select_bands = 'B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12'
else:
    select_bands = 'B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12'

# convert comma separated band list into a list
select_bands = [x for x in re.split(',', select_bands)]

if roi_lon_lat:
    roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [float(x) for x in re.split(',', roi_lon_lat)]
else:
    roi_lon1, roi_lat1, roi_lon2, roi_lat2 = -180, -90, 180, 90


raster_data = rasterio.open(data_file)
datasets = raster_data.subdatasets

for dsdesc in datasets:
    if '10m' in dsdesc:
        ds10 = rasterio.open(dsdesc)
    elif '20m' in dsdesc:
        ds20 = rasterio.open(dsdesc)
    elif '60m' in dsdesc:
        ds60 = rasterio.open(dsdesc)
    else:
        dsunknown = rasterio.open(dsdesc)


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
    return tmxmin,tmymin,tmxmax,tmymax, area


def to_xy(lon, lat):
    """
    This method gets the longitude and the latitude of a given point and projects it
    into pixel location in the new coordinate system.

    :param lon: The longitude of a chosen point
    :param lat: The longitude of a chosen point
    :return: The pixel location in the coordinate system of the input image
    """
    crs_wgs = proj.Proj(init='epsg:4326')
    crs_bng = proj.Proj(init='epsg:32639')
    xp, yp = proj.transform(crs_wgs, crs_bng, lon, lat)
    xp -= xoff
    yp -= yoff
    # matrix inversion
    det_inv = 1. / (a * e - d * b)
    x = (e * xp - b * yp) * det_inv
    y = (-d * xp + a * yp) * det_inv
    return (int(x), int(y))


if roi_x_y:
    roi_x1, roi_y1, roi_x2, roi_y2 = [float(x) for x in re.split(',', roi_x_y)]
    xmin, ymin, xmax, ymax, area = get_max_min(roi_x1, roi_y1, roi_x2, roi_y2)
elif not roi_lon_lat:
    xmin, ymin, xmax, ymax = (0, 0, ds10.width, ds10.height)
else:
    a, b,xoff, d, e, yoff = ds10.transform
    x1, y1 = to_xy(roi_lon1, roi_lat1)
    x2, y2 = to_xy(roi_lon2, roi_lat2)
    xmin, ymin, xmax, ymax, area = get_max_min(x1, y1, x2, y2)

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
    # Some HDR restrictions... ENVI band names should not include commas
    if output_file_format == 'ENVI' and ',' in description:
        pos = description.find(',')
        return description[:pos] + description[(pos + 1):]
    return description


if list_bands:
    logger.info("\n10m bands:")
    for b in range(0, ds10.count):
        logger.info("- " + validate_description(ds10.descriptions[b]))
    logger.info("\n20m bands:")
    for b in range(0, ds20.count):
        print("- " + validate_description(ds10.descriptions[b]))
    logger.info("\n60m bands:")
    for b in range(0, ds60.count):
        logger.info("- " + validate_description(ds10.descriptions[b]))
    logger.info("")


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
    validated_bands = []
    validated_indices = []
    validated_descriptions = defaultdict(str)
    for b in range(0, data.count):
        desc = validate_description(data.descriptions[b])
        name = get_band_short_name(desc)
        if name in select_bands:
            #logger.info(" " + name)
            select_bands.remove(name)
            validated_bands += [name]
            validated_indices += [b]
            validated_descriptions[name] = desc
    return validated_bands, validated_indices, validated_descriptions


logger.info("Selected 10m bands:")
validated_10m_bands, validated_10m_indices, dic_10m = validate(ds10)

logger.info("Selected 20m bands:")
validated_20m_bands, validated_20m_indices, dic_20m = validate(ds20)

logger.info("Selected 60m bands:")
validated_60m_bands, validated_60m_indices, dic_60m = validate(ds60)

validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}


if list_bands:
    sys.exit(0)

output_file = save_prefix + output_file
# Some HDR restrictions... ENVI file name should be the .bin, not the .hdr
if output_file_format == 'ENVI' and (output_file[-4:] == '.hdr' or output_file[-4:] == '.HDR'):
    output_file = output_file[:-4] + '.bin'


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


if run_60:
    data10 = data_final(ds10, validated_10m_indices, xmin, ymin, xmax, ymax, 1)
    data20 = data_final(ds20, validated_20m_indices, xmin // 2, ymin // 2, xmax // 2, ymax // 2, 1 // 2)
    data60 = data_final(ds60, validated_60m_indices, xmin // 6, ymin // 6, xmax // 6, ymax // 6, 1 // 6)
else:
    data10 = data_final(ds10, validated_10m_indices, xmin, ymin, xmax, ymax, 1)
    data20 = data_final(ds20, validated_20m_indices, xmin // 2, ymin // 2, xmax // 2, ymax // 2, 1 // 2)

if validated_60m_bands and validated_20m_bands and validated_10m_bands:
    logger.info("Super-resolving the 60m data into 10m bands")
    sr60 = DSen2_60(data10, data20, data60, deep=False)
else:
    sr60 = None

if validated_10m_bands and validated_20m_bands:
    logger.info("Super-resolving the 20m data into 10m bands")
    sr20 = DSen2_20(data10, data20, deep=False)
else:
    sr20 = None


if sr20 is None:
    logger.info("No super-resolution performed, exiting")
    sys.exit(0)


if sr60 is not None:
    sr = np.concatenate((sr20, sr60), axis=2)
    validated_sr_bands = validated_20m_bands + validated_60m_bands
else:
    sr = sr20
    validated_sr_bands = validated_20m_bands

if copy_original_bands:
    out_dims = data10.shape[2] + sr.shape[2]
else:
    out_dims = sr.shape[2]

logger.info("Writing")
logger.info(" the super-resolved bands in")
logger.info(output_file)


def update(data):
    """
    This method creates the proper georeferencing for the output image.
    :param data: The raster file for 10m resolution.

    """
    p = data.profile
    new_transform = p['transform'] * A.translation(xmin, ymin)
    p.update(dtype=rasterio.float32)
    p.update(driver='GTiff')
    p.update(width=data10.shape[1])
    p.update(height=data10.shape[0])
    p.update(count=out_dims)
    p.update(transform=new_transform)
    return p


profile = update(ds10)

with rasterio.open(os.path.join(OUTPUT_DIR, output_file), 'w' ,**profile) as ds10w:
    for bi, bn in enumerate(validated_sr_bands):
        ds10w.write(sr20[:,:,bi], indexes=bi+1)


if output_file_format == "npz":
    np.savez(output_file, bands=bands)
