import rasterio
import unittest
import glob
import os

from context import Superresolution

input_folder = '/tmp/input/'
data_folder = '*/*/MTD*.xml'
for file in glob.iglob(os.path.join(input_folder, data_folder), recursive=True):
    DATA_PATH = file


class SimpleFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.srr = Superresolution()
        self.ds10r, self.ds20r, self.ds60r, self.dsunknownr, self.output_jsonfile, self.output_name = \
            self.srr.get_data()

    def test_desc(self):
        validated_10m_indices_exm = [0, 1, 2, 3]
        validated_10m_bands_exm = ['B2', 'B3', 'B4', 'B8']
        validated_20m_indices_exm = [0, 1, 2, 3, 4, 5]
        validated_20m_bands_exm = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        validated_60m_indices_exm = [0, 1]
        validated_60m_bands_exm = ['B1', 'B9']
        validated_10m_bands, validated_10m_indices, dic_10m = self.srr.validate(self.ds10r)
        validated_20m_bands, validated_20m_indices, dic_20m = self.srr.validate(self.ds20r)
        validated_60m_bands, validated_60m_indices, dic_60m = self.srr.validate(self.ds60r)
        self.assertEqual(set(validated_10m_bands), set(validated_10m_bands_exm))
        self.assertEqual(set(validated_20m_bands), set(validated_20m_bands_exm))
        self.assertEqual(set(validated_60m_bands), set(validated_60m_bands_exm))
        self.assertEqual(validated_10m_indices, validated_10m_indices_exm)
        self.assertEqual(validated_20m_indices, validated_20m_indices_exm)
        self.assertEqual(validated_60m_indices, validated_60m_indices_exm)


class Superres(unittest.TestCase):
    def setUp(self):
        srr = Superresolution()
        srr.run()
        for out_file in glob.iglob(os.path.join('/tmp/output/', '*.tif'), recursive=True):
            output_image_path = out_file
        self.output_image = rasterio.open(output_image_path)

    def test_output_transform(self):
        self.assertEqual(self.output_image.transform[0], 10)
        self.assertEqual(self.output_image.transform[4], -10)

    def test_output_description(self):
        desc_exm = ('SR B5 (705 nm)', 'SR B6 (740 nm)', 'SR B7 (783 nm)', 'SR B8A (865 nm)', 'SR B11 (1610 nm)',
                    'SR B12 (2190 nm)', 'SR B1 (443 nm)', 'SR B9 (945 nm)')
        self.assertEqual(self.output_image.descriptions, desc_exm)

    def test_output_projection(self):
        crs_exm = 'epsg:32639'
        self.assertEqual(self.output_image.crs, crs_exm)


if __name__ == '__main__':
    unittest.main()
