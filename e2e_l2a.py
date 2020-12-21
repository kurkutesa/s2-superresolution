"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
# WARNING
# THIS E2E TEST WILL ONLY WORK IN GPU ENABLED MACHINES

from blockutils.e2e import E2ETest
from e2e import asserts

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
        e2e.add_gs_bucket("gs://floss-blocks-e2e-testing/e2e_s2_superresolution_l2a/*")
        e2e.asserts = asserts
        e2e.run()
    else:
        print("Skipping test...")
