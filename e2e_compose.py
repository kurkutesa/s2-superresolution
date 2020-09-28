"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid. Additionaly, save log.txt file with block log messages and memory consumption statistics.
Option to make use of the GPU if available.

Is able to work with different data sources other than the defaul e2e test data
usin the --data parameter, i.e.:

python3 e2e_compose.py --data gs://floss-blocks-e2e-testing/e2e_s2_superresolution_memory/*
python3 e2e_compose.py -gpu --data gs://floss-blocks-e2e-testing/e2e_s2_superresolution_memory/*


"""
from pathlib import Path
import shutil
import argparse
import subprocess

from e2e import assert_e2e
from blockutils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test block with docker-compose and memory logging facility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="GCS bucket link to folder with input data. i.e. gs://floss-blocks-e2e-testing/e2e_s2_superresolution/*",
    )
    parser.add_argument(
        "-c",
        "--docker-compose",
        type=str,
        default="docker-compose.yml",
        help="Path to YAML file defining compose.",
    )
    parser.add_argument(
        "-gpu", action="save_true", default=False, help="Use docker run gpu call.",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="log.txt",
        help="Path of log file. Will be overwriten if exists!",
    )

    _args = parser.parse_args()
    _is_data_default = _args.data is None
    if _is_data_default:
        _args.data = "gs://floss-blocks-e2e-testing/e2e_s2_superresolution/*"
        logger.info("No specified data paramater.")
        logger.info(f"Using default={_args.data}.")

    return _args, _is_data_default


def run_command(command):
    subprocess.run(command, shell=True, check=True)
    return True


if __name__ == "__main__":
    ARGS, IS_DATA_DEFAULT = parse_args()
    TESTNAME = "e2e_s2-superresolution"
    TEST_DIR = Path("/tmp") / TESTNAME
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_DIR = TEST_DIR / "input"
    OUTPUT_DIR = TEST_DIR / "output"

    LOG = Path(ARGS.log)
    DATA = ARGS.data
    COMPOSE = Path(ARGS.docker_compose)

    # Cleanup
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if not IS_DATA_DEFAULT and INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if LOG.exists():
        LOG.unlink()

    if not COMPOSE.exists():
        raise ValueError("Docker compose file %s does not exist." % str(COMPOSE))

    # Prepare input data
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            "gsutil -m cp -r %s %s/" % (DATA, INPUT_DIR), shell=True, check=True
        )

    DOCKER_CMD = (
        f"docker-compose -f {str(COMPOSE)} --compatibility up --build >> {str(LOG)}"
    )
    if DATA.gpu:
        DOCKER_CMD = (
            "docker run --runtime=nvidia --rm --mount \
        type=bind,src=/tmp/e2e_s2-superresolution/output,dst=/tmp/output \
        --mount type=bind,src=/tmp/e2e_s2-superresolution/input,dst=/tmp/input \
        s2-superresolution \
        >> %s"
            % (str(LOG),)
        )
    LOGGER_CMD = f"docker stats --no-stream >> {str(LOG)}"

    DOCKER_POPEN = subprocess.Popen(DOCKER_CMD, shell=True)
    while True:
        if DOCKER_POPEN.poll() is None:
            LOGGER_POPEN = subprocess.Popen(LOGGER_CMD, shell=True)
            LOGGER_POPEN.wait()
        else:
            break

    print("Exited with code %r" % DOCKER_POPEN.poll())
    assert DOCKER_POPEN.poll() == 0

    assert_e2e(TEST_DIR)
