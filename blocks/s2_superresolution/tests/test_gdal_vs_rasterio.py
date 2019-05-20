from osgeo import gdal
import rasterio
import unittest
import re
import sys
from collections import defaultdict


def validate_description(description):
    m = re.match("(.*?), central wavelength (\d+) nm", description)
    if m:
        return m.group(1) + " (" + m.group(2) + " nm)"
    return description

def get_band_short_name(description):
    if ',' in description:
        return description[:description.find(',')]
    if ' ' in description:
        return description[:description.find(' ')]
    return description[:3]


def validate(data):
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


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.ds10 = gdal.Open('/Users/nikoo.ekhtiari/Documents/s2-superresolution/10m.tiff')
        self.ds10r = rasterio.open('/Users/nikoo.ekhtiari/Documents/s2-superresolution/10m.tiff')

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


    def test_array(self):
        data10 = self.ds10.ReadAsArray()
        data10r = self.ds10r.read()
        self.assertEqual(data10.all(), data10r.all())

    def test_desc(self):
        validated_10m_indices_exm = [0, 1, 2, 3]
        validated_10m_bands_exm = ['B2', 'B3', 'B4', 'B8']
        validated_10m_bands, validated_10m_indices, dic_10m = validate(self.ds10r)
        self.assertEqual(set(validated_10m_bands), set(validated_10m_bands_exm))
        self.assertEqual(validated_10m_indices, validated_10m_indices_exm)


if __name__ == '__main__':
    unittest.main()
