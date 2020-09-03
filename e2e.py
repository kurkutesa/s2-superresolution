"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
from pathlib import Path
import os

import geojson
import rasterio

# WARNING
# THIS E2E TEST WILL ONLY WORK IN GPU ENABLED MACHINES
def assert_e2e(test_dir):
    # Print out bbox of one tile
    geojson_path = test_dir / "output" / "data.json"

    with open(str(geojson_path)) as f:
        feature_collection = geojson.load(f)

    print(feature_collection.features[0].bbox)

    output = (
        test_dir
        / "output"
        / Path(feature_collection.features[0].properties["up42.data_path"])
    )

    print(output)

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


def setup(testname):
    test_dir = Path("/tmp") / testname
    test_dir.mkdir(parents=True, exist_ok=True)
    input_dir = test_dir / "input"
    files_to_delete = Path(test_dir / "output").glob("*")
    for file_path in files_to_delete:
        file_path.unlink()

    # Prepare input data
    if not os.path.isdir(input_dir):
        input_dir.mkdir(parents=True, exist_ok=True)
        os.system(
            "gsutil -m cp -r gs://floss-blocks-e2e-testing/e2e_s2_superresolution/* %s"
            % input_dir
        )

    run_cmd = (
        """docker run -v %s:/tmp \
                 -e 'UP42_TASK_PARAMETERS={"bbox": [12.211, 52.291, 12.513, 52.521], "clip_to_aoi": true, \
                 "copy_original_bands": false}' \
                 -it s2-superresolution"""
        % test_dir
    )

    return run_cmd, test_dir


if __name__ == "__main__":
    TESTNAME = "e2e_s2-superresolution"
    RUN_CMD, TEST_DIR = setup(TESTNAME)

    os.system(RUN_CMD)

    assert_e2e(TEST_DIR)
