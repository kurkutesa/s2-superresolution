from osgeo import gdal
import rasterio
import unittest
import re
import sys
from collections import defaultdict
import glob
import os

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
    validated_bands = []
    validated_indices = []
    validated_descriptions = defaultdict(str)
    select_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    for b in range(0, data.count):
        desc = validate_description(data.descriptions[b])
        name = get_band_short_name(desc)
        if name in select_bands:
            sys.stdout.write(" " + name)
            select_bands.remove(name)
            validated_bands += [name]
            validated_indices += [b]
            validated_descriptions[name] = desc
    return validated_bands, validated_indices, validated_descriptions


input_folder = '/tmp/input/'
data_folder = '*/MTD*.xml'
for file in glob.iglob(os.path.join(input_folder, data_folder), recursive=True):
    DATA_PATH = file


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        gdal_data = gdal.Open(DATA_PATH)
        raster_data = rasterio.open(DATA_PATH)
        gdal_datasets = gdal_data.GetSubDatasets()
        raster_datasets = raster_data.subdatasets
        self.ds10 = gdal.Open(gdal_datasets[0][0])
        self.ds10r = rasterio.open(raster_datasets[0])

    def test_description(self):
        d = self.ds10r.descriptions
        self.assertEqual(self.ds10.GetRasterBand(1).GetDescription(), d[0])

    def test_rastersize(self):
        self.assertEqual(self.ds10.RasterXSize, self.ds10r.width)
        self.assertEqual(self.ds10.RasterYSize, self.ds10r.height)

    def test_transfrom(self):
        t = self.ds10.GetGeoTransform()
        tr = self.ds10r.transform
        tr = tuple(tr)[:-3]
        self.assertEqual(set(t), set(tr))

    def test_projection(self):
        p = self.ds10.GetProjection()
        pr = self.ds10r.crs.wkt
        self.assertEqual(p, pr)

    def test_desc(self):
        validated_10m_indices_exm = [0, 1, 2, 3]
        validated_10m_bands_exm = ['B2', 'B3', 'B4', 'B8']
        # validated_20m_indices_exm = [0, 1, 2, 3, 4, 5]
        # validated_20m_bands_exm = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        # validated_60m_indices_exm = [0, 1, 2]
        # validated_60m_bands_exm = ['B1', 'B9', 'B10']
        validated_10m_bands, validated_10m_indices, dic_10m = validate(self.ds10r)
        self.assertEqual(set(validated_10m_bands), set(validated_10m_bands_exm))
        self.assertEqual(validated_10m_indices, validated_10m_indices_exm)


if __name__ == '__main__':
    unittest.main()
