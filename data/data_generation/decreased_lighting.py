import cv2
import numpy as np
import math
import random
import tifffile
from os import listdir
from os.path import isfile, join

# Source data directory
source_dir = 'datasets/control/'
dest_dir = 'datasets/decreased_lighting/'

print("\n*** ArDVRC BACKGROUND SHAPES GENERATOR ***\n")
print("This script takes a dataset and reduces contrast.\n")
print("The input dataset is %s. Make sure this is correct!\n" % source_dir)
print("WARNING: THE GENERATED FILES TAKE UP A LOT OF HARDDRIVE SPACE. Make sure your computer has enough space!\n")

prompt = "WARNING: This is expected to take up at least the size of your input dataset. Do you want to continue? (y/n) "
response = input(prompt)
print()
if response.lower() == 'n':
    print("Exiting script...")
    exit()
elif response.lower() != 'y':
    print("ERROR: Unexpected response. Exiting...")
    exit()

import os
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
else:
    print('ERROR: The output folder %s already exists. Move, rename or delete this directory and run this program again.' % dest_dir)
    exit()

scaling = .5
filenames = [f for f in listdir(source_dir) if (isfile(join(source_dir, f)) and f.endswith('.tif'))]

print("Writing images...")
# Source folder
for f in filenames:
    with tifffile.TiffFile(join(source_dir, f)) as tif:
        a = tif.asarray()
        a = a*scaling
        a = a.astype(np.uint8)
        meta = tif.shaped_metadata[0]

        # Remove shape from meta since it will be re-added
        meta.pop('shape',None)
          
        tifffile.imwrite(join(dest_dir,f), a, metadata=meta)
        #cv2.imshow('out', a)
        #k = cv2.waitKey(0)
        #if k == ord('q'):
        #	exit()

print("Finished writing images, exiting...")