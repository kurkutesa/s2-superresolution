"""
This module creates a synthetic image.
"""
import uuid
from pathlib import Path

import rasterio
from rasterio import crs
from rasterio.transform import from_origin
import skimage.draw
from scipy import signal
import numpy as np

from predict.helper import get_logger

LOGGER = get_logger(__name__)

# Define standard land cover classes and their standard stats wrt radiance for a 4-band product
LC_CLASSES = {"water": {"avg": [170, 300, 450, 150], "std": [10, 10, 10, 10]},
              "bare_ground": {"avg": [600, 600, 600, 900], "std": [100, 100, 100, 100]},
              "built_up": {"avg": [800, 800, 800, 1000], "std": [150, 150, 150, 150]},
              "forest": {"avg": [250, 450, 450, 1300], "std": [30, 30, 30, 30]},
              "non-forest_vegetation": {"avg": [300, 500, 500, 1100], "std": [100, 100, 100, 100]}}


class SyntheticImage:
    # pylint: disable=E1137
    # pylint: disable=R0913
    """
    Create synthetic GeoTIFF test image. An image created this
    way cannot recreate all characteristics of a real geospatial
    image, but if cleverly created can avoid having to use golden files for a
    long list of cases.

    :param xsize: number of pixels in x-direction
    :param ysize: number of pixels in y-direction
    :param num_bands: number of image bands
    :param data_type: rasterio datatype as string
    :param out_dir: Path where the image should be created (default ".")
    :param coord_ref_sys: EPSG identifier of used coordinate reference system (default 3837)
    :param nodata_fill: number of no data pixels to set in top left image (in x and y).
    :return: tuple(Path to the output image, numpy array of image values)
    """

    def __init__(self, xsize: int, ysize: int, num_bands: int, data_type: str,
                 out_dir: Path = Path('.'), coord_ref_sys: int = 3857,
                 nodata_fill: int = 0):

        self.xsize = xsize
        self.ysize = ysize
        self.num_bands = num_bands
        self.data_type = data_type
        self.out_dir = out_dir
        self.crs = coord_ref_sys
        self.nodata_fill = nodata_fill

    def add_img_pattern(self, seed):
        """
        Simulate a five classes optical image
        """
        if seed is not None:
            np.random.seed(seed)

        image, _ = skimage.draw.random_shapes((self.ysize, self.xsize), max_shapes=50,
                                              min_shapes=25, multichannel=False,
                                              allow_overlap=True, random_seed=seed)
        # Assign shape values to output classes
        image[image < 55] = 1
        image[(image >= 55) & (image < 105)] = 2
        image[(image >= 105) & (image < 155)] = 3
        image[(image >= 155) & (image < 205)] = 4
        image[(image >= 205) & (image <= 255)] = 5

        # Create bands having relevant values for all output classes
        bands = []
        band_idx = 0
        while band_idx < self.num_bands:
            data_ar = np.zeros_like(image, dtype=self.data_type)

            for class_idx, lc_class in enumerate(LC_CLASSES.values(), 1):
                # Add Gaussian noise
                mask_ar = np.random.normal(lc_class["avg"][band_idx],
                                           lc_class["std"][band_idx],
                                           image.shape)
                data_ar[image == class_idx] = mask_ar[image == class_idx] #pylint: disable=unsubscriptable-object
            # Apply median filter to simulate spatial autocorrelation
            data_ar = (signal.medfilt(data_ar)).astype(self.data_type)
            data_ar = np.clip(data_ar, 1, None)
            bands.append(data_ar)
            band_idx += 1

        return bands

    def create(self, pix_width, pix_height, valid_desc, seed: int = None) -> tuple:
        """
        This methods create a synthetic image.
        """
        band_list = self.add_img_pattern(seed)

        filepath = self.out_dir.joinpath(str(uuid.uuid4()) + ".tif")

        transform = from_origin(1470996, 6914001, pix_width, pix_height)

        with rasterio.open(filepath, 'w', driver='GTiff', height=self.ysize, width=self.xsize,
                           count=self.num_bands, dtype=str(band_list[0].dtype),
                           crs=crs.CRS.from_epsg(self.crs),
                           transform=transform) as out_img:
            for band_id, layer in enumerate(band_list):
                layer[0:self.nodata_fill, 0:self.nodata_fill] = 0
                out_img.write_band(band_id + 1, layer)
                out_img.set_band_description(band_id + 1, valid_desc[band_id])

        return filepath, np.array(band_list)
