"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
import os

# WARNING
# THIS E2E TEST WILL ONLY WORK IN GPU ENABLED MACHINES

from e2e import assert_e2e, setup

if __name__ == "__main__":
    TESTNAME = "e2e_s2-superresolution-l2a"
    RUN_CMD, TEST_DIR = setup(TESTNAME)

    os.system(RUN_CMD)

    assert_e2e(TEST_DIR)
