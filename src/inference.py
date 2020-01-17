import sys
import os
import gc

import numpy as np

from s2_tiles_supres import Superresolution
from supres import dsen2_20, dsen2_60


from helper import get_logger, save_result, load_params

LOGGER = get_logger(__name__)


class SuperresolutionProcess(Superresolution):
    # pylint: disable=too-many-locals
    def process(self, path_to_input_img, path_to_output_img):
        data_list = self.get_data(path_to_input_img)

        for dsdesc in data_list:
            if "10m" in dsdesc:
                xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(dsdesc)
                LOGGER.info("Selected pixel region:")
                LOGGER.info("xmin = %s", xmin)
                LOGGER.info("ymin = %s", ymin)
                LOGGER.info("xmax = %s", xmax)
                LOGGER.info("ymax = %s", ymax)
                LOGGER.info("The area of selected region = %s", interest_area)
                if xmax < xmin or ymax < ymin:
                    LOGGER.info("Invalid region of interest / UTM Zone combination")
                    sys.exit(0)

        for dsdesc in data_list:
            if "10m" in dsdesc:
                LOGGER.info("Selected 10m bands:")
                validated_10m_bands, validated_10m_indices, dic_10m = self.validate(
                    dsdesc
                )
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1
                )
            if "20m" in dsdesc:
                LOGGER.info("Selected 20m bands:")
                validated_20m_bands, validated_20m_indices, dic_20m = self.validate(
                    dsdesc
                )
                data20 = self.data_final(
                    dsdesc,
                    validated_20m_indices,
                    xmin // 2,
                    ymin // 2,
                    xmax // 2,
                    ymax // 2,
                    1 // 2,
                )
            if "60m" in dsdesc:
                LOGGER.info("Selected 60m bands:")
                validated_60m_bands, validated_60m_indices, dic_60m = self.validate(
                    dsdesc
                )
                data60 = self.data_final(
                    dsdesc,
                    validated_60m_indices,
                    xmin // 6,
                    ymin // 6,
                    xmax // 6,
                    ymax // 6,
                    1 // 6,
                )

        validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}

        if validated_60m_bands and validated_20m_bands and validated_10m_bands:
            LOGGER.info("Super-resolving the 60m data into 10m bands")
            sr60 = dsen2_60(data10, data20, data60, deep=False)
            LOGGER.info("Super-resolving the 20m data into 10m bands")
            sr20 = dsen2_20(data10, data20, deep=False)
        else:
            LOGGER.info("No super-resolution performed, exiting")
            sys.exit(0)

        if self.params["copy_original_bands"]:
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
    SuperresolutionProcess(PARAMS).process(sys.argv[1], sys.argv[2])
