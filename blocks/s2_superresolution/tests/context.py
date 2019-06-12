import os
import sys
# sys.path.insert('../src')

# Path hacks to make the code available for testing
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the required classes and functions
from predict.s2_tiles_supres import Superresolution #pylint: disable=unused-import,wrong-import-position
from predict.helper import LOG_FORMAT, get_logger, load_params, ensure_data_directories_exist #pylint: disable=unused-import,wrong-import-position,line-too-long