import sys
import os
import gc

import numpy as np
import rasterio

from blockutils.logging import get_logger
from blockutils.common import load_params

from s2_tiles_supres import Superresolution
from supres import dsen2_20, dsen2_60

LOGGER = get_logger(__name__)


# pylint: disable-msg=too-many-arguments
def save_result(
    model_output, output_bands, valid_desc, output_profile, image_name,
):
    """
    This method saves the feature collection meta data and the
    image with high resolution for desired bands to the provided location.

    Args:
        model_output: The high resolution image.
        output_bands: The associated bands for the output image.
        valid_desc: The valid description of the existing bands.
        output_profile: The georeferencing for the output image.
        output_features: The meta data for the output image.
        image_name: The name of the output image.
    """

    with rasterio.open(image_name, "w", **output_profile) as d_s:
        for b_i, b_n in enumerate(output_bands):
            d_s.write(model_output[:, :, b_i], indexes=b_i + 1)
            d_s.set_band_description(b_i + 1, "SR " + valid_desc[b_n])


class SuperresolutionProcess(Superresolution):
    # pylint: disable=too-many-locals
    @staticmethod
    def check_size(dims):
        xmin, ymin, xmax, ymax = dims
        if xmax < xmin or ymax < ymin:
            LOGGER.error("Invalid region of interest / UTM Zone combination")
            sys.exit(1)

        if (xmax - xmin) < 192 or (ymax - ymin) < 192:
            LOGGER.error(
                "AOI too small. Try again with a larger AOI (minimum pixel width or heigh of 192)"
            )
            sys.exit(1)

    def start(self, path_to_input_img, path_to_output_img):
        data_list, image_level = self.get_data(path_to_input_img)

        for dsdesc in data_list:
            if "10m" in dsdesc:
                if self.params.__dict__["clip_to_aoi"]:
                    xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(
                        dsdesc
                    )
                else:
                    # Get the pixel bounds of the full scene
                    xmin, ymin, xmax, ymax, interest_area = self.get_max_min(
                        0, 0, 20000, 20000, dsdesc
                    )
                LOGGER.info("Selected pixel region:")
                LOGGER.info("xmin = {xmin}")
                LOGGER.info("ymin = {ymin}")
                LOGGER.info("xmax = %s", xmax)
                LOGGER.info("ymax = %s", ymax)
                LOGGER.info("The area of selected region = %s", interest_area)
            self.check_size(dims=(xmin, ymin, xmax, ymax))

        for dsdesc in data_list:
            if "10m" in dsdesc:
                LOGGER.info("Selected 10m bands:")
                validated_10m_bands, validated_10m_indices, dic_10m = self.validate(
                    dsdesc
                )
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1, 1
                )
            if "20m" in dsdesc:
                LOGGER.info("Selected 20m bands:")
                validated_20m_bands, validated_20m_indices, dic_20m = self.validate(
                    dsdesc
                )
                data20 = self.data_final(
                    dsdesc, validated_20m_indices, xmin, ymin, xmax, ymax, 1, 2
                )
            if "60m" in dsdesc:
                LOGGER.info("Selected 60m bands:")
                validated_60m_bands, validated_60m_indices, dic_60m = self.validate(
                    dsdesc
                )
                data60 = self.data_final(
                    dsdesc, validated_60m_indices, xmin, ymin, xmax, ymax, 1, 6
                )

        validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}

        if validated_60m_bands and validated_20m_bands and validated_10m_bands:
            LOGGER.info("Super-resolving the 60m data into 10m bands")
            sr60 = dsen2_60(data10, data20, data60, image_level)
            LOGGER.info("Super-resolving the 20m data into 10m bands")
            sr20 = dsen2_20(data10, data20, image_level)
        else:
            LOGGER.info("No super-resolution performed, exiting")
            sys.exit(0)

        if self.params.__dict__["copy_original_bands"]:
            sr_final = np.concatenate((data10, sr20, sr60), axis=2)
            validated_sr_final_bands = (
                validated_10m_bands + validated_20m_bands + validated_60m_bands
            )
        else:
            sr_final = np.concatenate((sr20, sr60), axis=2)
            validated_sr_final_bands = validated_20m_bands + validated_60m_bands

        for dsdesc in data_list:
            if "10m" in dsdesc:
                p_r = self.update(dsdesc, data10.shape, sr_final, xmin, ymin)
        filename = os.path.join(self.output_dir, path_to_output_img)

        LOGGER.info("Now writing the super-resolved bands")
        save_result(
            sr_final,
            validated_sr_final_bands,
            validated_descriptions_all,
            p_r,
            filename,
        )
        del sr_final
        LOGGER.info("This is for releasing memory: %s", gc.collect())
        LOGGER.info("Writing the super-resolved bands is finished.")


if __name__ == "__main__":
    PARAMS = load_params()
    SuperresolutionProcess(PARAMS).start(sys.argv[1], sys.argv[2])
