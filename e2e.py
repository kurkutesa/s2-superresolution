"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
from pathlib import Path
import os

import geojson
import rasterio

if __name__ == "__main__":
    TESTNAME = "e2e_s2-superresolution"
    TEST_DIR = Path("/tmp") / TESTNAME
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR = TEST_DIR / "input"
    FILES_TO_DELETE = Path(TEST_DIR / "output").glob("*")
    for file_path in FILES_TO_DELETE:
        file_path.unlink()

    # Prepare input data
    if not os.path.isdir(INPUT_DIR):
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.system(
            "gsutil -m cp -r gs://floss-blocks-e2e-testing/e2e_s2_superresolution/* %s"
            % INPUT_DIR
        )

    RUN_CMD = (
        """docker run -v %s:/tmp \
                 -e 'UP42_TASK_PARAMETERS={"roi_x_y": [5000, 5000, 5250, 5250], \
                 "copy_original_bands": false}' \
                 -it s2-superresolution"""
        % TEST_DIR
    )

    os.system(RUN_CMD)

    # Print out bbox of one tile
    GEOJSON_PATH = TEST_DIR / "output" / "data.json"

    with open(str(GEOJSON_PATH)) as f:
        FEATURE_COLLECTION = geojson.load(f)

    print(FEATURE_COLLECTION.features[0].bbox)

    OUTPUT = (
        TEST_DIR
        / "output"
        / Path(FEATURE_COLLECTION.features[0].properties["up42.data.aoiclipped"])
    )

    print(OUTPUT)

    assert OUTPUT.exists()

    # Check whether the outcome image has the correct 10m resolution for all the spectral bands.
    OUTPUT_IMAGE = rasterio.open(OUTPUT)
    assert OUTPUT_IMAGE.transform[0] == 10
    assert OUTPUT_IMAGE.transform[4] == -10

    DESC_EXM = (
        "SR B5 (705 nm)",
        "SR B6 (740 nm)",
        "SR B7 (783 nm)",
        "SR B8A (865 nm)",
        "SR B11 (1610 nm)",
        "SR B12 (2190 nm)",
        "SR B1 (443 nm)",
        "SR B9 (945 nm)",
    )
    assert OUTPUT_IMAGE.descriptions == DESC_EXM

    # Check whether the outcome image has the correct georeference.
    CRS_EXM = {"init": "epsg:32633"}
    assert OUTPUT_IMAGE.crs.to_dict() == CRS_EXM
