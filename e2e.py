"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
from pathlib import Path

import geojson
import rasterio

from blockutils.e2e import E2ETest

# WARNING
# THIS E2E TEST WILL ONLY WORK IN GPU ENABLED MACHINES

# Disable unused params for assert
# pylint: disable=unused-argument
def asserts(input_dir: Path, output_dir: Path, quicklook_dir: Path, logger):
    # Print out bbox of one tile
    geojson_path = output_dir / "data.json"

    with open(str(geojson_path)) as f:
        feature_collection = geojson.load(f)

    logger.info(feature_collection.features[0].bbox)

    output = output_dir / feature_collection.features[0].properties["up42.data_path"]

    logger.info(output)

    assert output.exists()

    # Check whether the outcome image has the correct 10m resolution for all the spectral bands.
    with rasterio.open(output) as output_image:
        assert output_image.transform[0] == 10
        assert output_image.transform[4] == -10

        desc_exm = (
            "SR B5 (705 nm)",
            "SR B6 (740 nm)",
            "SR B7 (783 nm)",
            "SR B8A (865 nm)",
            "SR B11 (1610 nm)",
            "SR B12 (2190 nm)",
            "SR B1 (443 nm)",
            "SR B9 (945 nm)",
        )
        assert output_image.descriptions == desc_exm

        # Check whether the outcome image has the correct georeference.
        crs_exm = {"init": "epsg:32633"}
        assert output_image.crs.to_dict() == crs_exm


if __name__ == "__main__":
    e2e = E2ETest("s2-superresolution")
    if not e2e.in_ci:
        e2e.add_parameters(
            {
                "bbox": [12.211, 52.291, 12.212, 52.290],
                "clip_to_aoi": True,
                "copy_original_bands": False,
            }
        )
        e2e.add_gs_bucket("gs://floss-blocks-e2e-testing/e2e_s2_superresolution/*")
        e2e.asserts = asserts
        e2e.run()
    else:
        print("Skipping test...")
