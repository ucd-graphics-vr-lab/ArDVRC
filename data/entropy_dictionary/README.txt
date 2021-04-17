ArDVRC High-Entropy Dictionary Generation Script

This folder contains a script to generate a dictionary that is based on ArUco's 4X4 dictionaries, but
is optimized to have a high number of corners (or high entropy). It also contains DICT_4X4_64_ENTROPY.npy,
which is the output of the script. WARNING: RUNNING THE SCRIPT WILL OVERWRITE THE NPY FILE.

PYTHON VERSION: 3.x
REQUIRED PYTHON MODULES (install with pip):
	opencv-contrib-python, opencv-python, numpy
TESTED ON: Windows (although this should be cross-platform in theory, will confirm later)

USAGE:
	1) The provided NumPy file, DICT_4X4_64_ENTROPY.npy, containing 64 4X4 high-entropy markers,
	   can be used in ArDVRC or ArUco Python or C++ programs as-is to replace an ArUco 4X4 dictionary 
	   (see sample applications and data generation scripts for examples of usage in both languages).

	2) The NumPy file was created using generate_entropy_dictionary.py. You can generate your own 
	   dictionary by running this script, and can change the parameters in the script to customize 
	   the dictionary to your needs. For example, if you wanted more markers in your dictionary, 
	   reduce the corner or hamming distance thresholds, or change the starting dictionary from
	   DICT_4X4_100 to a larger 4X4 dictionary.

TODO:
	1) It might be nice to expand this to dictionaries for 5X5, 6X6 and 7X7 in the future.
