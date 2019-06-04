import sys
sys.path.append('/block/src')
from predict.s2_tiles_supres_copy import Superresolution

if __name__ == "__main__":
	Superresolution.run()