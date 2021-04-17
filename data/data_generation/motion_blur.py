import cv2
import numpy as np
import math
import random
import tifffile
from os import listdir
from os.path import isfile, join
import os

# Modify this to the location of the dataset to which you want to add motion blur
source_dir = "datasets/control/"

print("\n*** ArDVRC MOTION BLUR GENERATOR ***\n")
print("This script takes a dataset and adds motion blur.\n")
print("The input dataset is %s. Make sure this is correct!\n" % source_dir)
print("WARNING: THE GENERATED FILES TAKE UP A LOT OF HARDDRIVE SPACE. Make sure your computer has enough space!\n")

kernel_size = 0

try:
    kernel_size = int(input("Enter blur kernel size (2-50): "))
    print()
except ValueError:
    print("ERROR: Invalid value for kernel size, exiting...")
    exit()

if kernel_size < 2 or kernel_size > 50:
    print("ERROR: Invalid range for blur kernel size, exiting...")
    exit()

blur_direction = None
bd = input("Enter vertical or horizontal blur direction (v/h): ")
print()
if (bd.lower() == 'h'):
    blur_direction = 'horizontal'
elif (bd.lower() == 'v'):
    blur_direction = 'vertical'
else:
    print("ERROR: Unexpected response. Exiting...")
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

# Create the vertical kernel. 
kernel_v = np.zeros((kernel_size, kernel_size)) 
  
# Create a copy of the same for creating the horizontal kernel. 
kernel_h = np.copy(kernel_v) 
  
# Fill the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 

# Normalize. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 

# Source data directory
dest_dir = 'datasets/motion_blur_'+ blur_direction + '_kernel_%i' % kernel_size
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
else:
    print('ERROR: The output folder %s already exists. Move, rename or delete this directory and run this program again.' % dest_dir)
    exit()

filenames = [f for f in listdir(source_dir) if (isfile(join(source_dir, f)) and f.endswith('.tif'))]

print("Writing images...")
# Source folder
for f in filenames:
    with tifffile.TiffFile(join(source_dir, f)) as tif:
        a = tif.asarray()
        meta = tif.shaped_metadata[0]

        # Remove shape from meta since it will be re-added
        meta.pop('shape',None)
          
        if blur_direction == 'vertical':
            # Apply the vertical kernel. 
            vertical_mb = cv2.filter2D(a, -1, kernel_v)
            tifffile.imwrite(join(dest_dir,f), vertical_mb, metadata=meta)

        elif blur_direction == 'horizontal': 
          
            # Apply the horizontal kernel. 
            horizontal_mb = cv2.filter2D(a, -1, kernel_h)
            tifffile.imwrite(join(dest_dir,f), horizontal_mb, metadata=meta)

print("Finished writing images, exiting...")