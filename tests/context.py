"""
This module is used in test_s2_tiles_supres script.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))


# pylint: disable=unused-import,wrong-import-position
from s2_tiles_supres import Superresolution
from supres import dsen2_60, dsen2_20, BatchGenerator
import patches
