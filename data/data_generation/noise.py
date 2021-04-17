import cv2
import numpy as np
import math
import random
import tifffile
from os import listdir
from os.path import isfile, join

# Modify source_dir to the location of the dataset to which you want to add noise
source_dir = "datasets/control/"

# Modify the name of dest_dir to change where the files will be written
dest_dir = 'datasets/noise/'

print("\n*** ArDVRC NOISE GENERATOR ***\n")
print("This script takes a dataset and adds noise.\n")
print("The input dataset is %s. Make sure this is correct!\n" % source_dir)
print("WARNING: THE GENERATED FILES TAKE UP A LOT OF HARDDRIVE SPACE. Make sure your computer has enough space!\n")

import os
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
else:
    print('ERROR: The output folder %s already exists. Move, rename or delete this directory and run this program again.' % dest_dir)
    exit()

prompt = "WARNING: This is expected to take up approximately the size of your input dataset. Do you want to continue? (y/n) "
response = input(prompt)
print()
if response.lower() == 'n':
    print("Exiting script...")
    exit()
elif response.lower() != 'y':
    print("ERROR: Unexpected response. Exiting...")
    exit()

filenames = [f for f in listdir(source_dir) if (isfile(join(source_dir, f)) and f.endswith('.tif'))]

print("Writing images...")
for f in filenames:
    with tifffile.TiffFile(join(source_dir, f)) as tif:
        a = tif.asarray()
        rand = a.copy()
        rand = cv2.randu(rand,0,255);

        a = cv2.addWeighted(a, 0.5, rand, 0.5, 0.0)
        meta = tif.shaped_metadata[0]

        # Remove shape from meta since it will be re-added
        meta.pop('shape',None)
          
        tifffile.imwrite(join(dest_dir,f), a, metadata=meta)

print("Finished writing images, exiting...")