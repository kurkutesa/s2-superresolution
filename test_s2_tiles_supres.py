from osgeo import gdal, osr
import rasterio
import unittest


class TestStringMethods(unittest.TestCase):
	def setUp(self):
		self.ds10 = gdal.Open('10m.tiff')
		self.ds10r = rasterio.open('10m.tiff')

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


	def test_array(self):
		data10 = self.ds10.ReadAsArray()
		data10r = self.ds10r.read()
		self.assertEqual(data10.all(), data10r.all())


if __name__ == '__main__':
	unittest.main()
