'''
data_generator.py

Description: Generates control dataset for ArDVRC and ArUco tests.
    Script outputs TIFF images of single markers of random pose and size
    (within constraints). User inputs the size of the control dataset.

'''
import cv2
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R
import tifffile

print("\n*** ArDVRC DATA GENERATOR ***\n")
print("This script generates TIFF images of single ArUco markers (using a custom high-entropy dictionary)\n")
print("WARNING: THE GENERATED FILES TAKE UP A LOT OF HARDDRIVE SPACE. Make sure your computer has enough space!\n")

# Simulating the ZED camera
# ZED image size for HD720 (what we will use in the actual program)

# Image size. Center point (cx, cy) for camera matrix taken from half the image size
image_size = (1280,720)

# Focal length info taken from ZED's website: 
# https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
focal_length = 700

scale = 1

focal_length = int(focal_length*scale)
image_size = (int(image_size[0]*scale), int(image_size[1]*scale))

# Camera matrix (imitating ZED camera)
zed_camera_matrix = np.array([[focal_length, 0.0, image_size[0]/2],[0.0, focal_length, image_size[1]/2],[0.0, 0.0, 1.0]])

#-------- SETTINGS --------#

# Whether or not to generate images with corners that fall outside the bounds of the image
# (Will generally fail for ArUco's detection method)
ALLOW_CORNERS_OUTSIDE_IMAGE = False

# Min and max rotation values (in degrees)
MIN_R_X = -45   # Min rotation about X axis
MAX_R_X = 45    # Max rotation about X axis
MIN_R_Y = -45   # Min rotation about Y axis
MAX_R_Y = 45    # Max rotation about Y axis
MIN_R_Z = -45   # Min rotation about Z axis
MAX_R_Z = 45    # Max rotation about Z axis

# Min and max translation values
# NOTE: Decided not to translate in Z direction in favor of just changing the marker size
# Play with these to change the possible pose of the marker
# These parameters keep the marker pretty well-constrained, were chosen through experimentation
MIN_T_X = -image_size[0]    # Min X translation
MAX_T_X = image_size[0]     # Max X translation
MIN_T_Y = -500              # Min Y translation
MAX_T_Y = 500               # Max Y translation
T_Z = 2000                  # Constant for X translation (modify marker size instead)

# Marker size
MIN_ID_SIZE = 250
MAX_ID_SIZE = 1500

# ID value ranges
MIN_ID_VAL = 0
MAX_ID_VAL = 63

# Start with standard ArUco dictionary. Make sure it is a 4X4, the last number doesn't matter
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

# Replace DICT_4X4_50's bytes list with our custom high-entropy dictionary
aruco_dict.bytesList = np.load('../entropy_dictionary/DICT_4X4_64_ENTROPY.npy')

#------------------#
n_images = 0
n_output_images = 0
#folder = 'output/'

try:
    n_output_images = int(input("Number of output images (range: 0-1000000): "))
    print()
    #n_output_images = 10000 # Set this higher for final dataset
except ValueError:
    print("ERROR: Invalid number of output images, exiting...")
    exit()

# Make sure number of images is within range
if n_output_images < 0 or n_output_images > 1000000:
    print("ERROR: Number of output images not within range, exiting...")
    exit()

prompt = "WARNING: This is expected to take up approximately %i bytes of space. Do you want to continue? (y/n) " % (image_size[0]*image_size[1]*n_output_images)
response = input(prompt)
print()
if response.lower() == 'n':
    print("Exiting script...")
    exit()
elif response.lower() != 'y':
    print("ERROR: Unexpected response. Exiting...")
    exit()

folder = "datasets/control/"

import os
if not os.path.exists(folder):
    print("Creating directory %s..." % folder)
    os.makedirs(folder)
else:
    print('ERROR: The output folder %s already exists. Move, rename or delete this directory and run this program again.' % folder)
    exit()

print("Writing images...")

while (n_images < n_output_images):

    # Random ID size (in pixels)
    id_size = random.randrange(MIN_ID_SIZE,MAX_ID_SIZE)

    # Random ID value
    id_val = random.randrange(MIN_ID_VAL, MAX_ID_VAL)

    # 2D plane in 3D space (depth==0)
    plane_points = np.array([[int(-id_size/2),int(-id_size/2),0], [int(id_size/2),int(-id_size/2),0], [int(id_size/2),int(id_size/2),0], [int(-id_size/2),int(id_size/2),0]], dtype=np.float32)

    # Calculate a random rotation
    r_x = random.randrange(MIN_R_X, MAX_R_X)
    r_y = random.randrange(MIN_R_Y, MAX_R_Y)
    r_z = random.randrange(MIN_R_Z, MAX_R_Z)

    # Random translation
    t_x = random.randrange(MIN_T_X, MAX_T_X)
    t_y = random.randrange(MIN_T_Y, MAX_T_Y)
    t_z = T_Z

    # Convert from Euler rotations to Axis-Angle (Rodrigues) vector
    rvec = R.from_euler('XYZ', np.array([r_x,r_y,r_z]), degrees=True).as_rotvec()

    # Translation vector
    tvec = np.array([t_x,t_y,t_z], np.float32)

    # Project 3D points to 2D screen space
    imagePoints, j = cv2.projectPoints(plane_points, rvec, tvec, zed_camera_matrix, np.zeros((5,1),np.float32))

    # Check that everything is within the bounds (if desired)
    if not ALLOW_CORNERS_OUTSIDE_IMAGE:
        valid = True
        for pt in imagePoints:
            if pt[0][0] < 0 or pt[0][0] > image_size[0]:
                valid = False
                continue
            if pt[0][1] < 0 or pt[0][1] > image_size[1]:
                valid = False
                continue
        if not valid:
            continue

    # 3D points for perspective transform
    id_points = np.array([[0,0,0], [id_size,0,0], [id_size,id_size,0], [0,id_size,0]], dtype=np.float32)

    M0 = cv2.getPerspectiveTransform(id_points[:,:2].reshape((1,4,2)), imagePoints.reshape((1,4,2)))

    # ID image
    img = cv2.aruco.drawMarker(aruco_dict, id_val, id_size)
    img = cv2.warpPerspective(img, M0, (int(image_size[0]), int(image_size[1])))

    dst = np.ones((int(image_size[1]), int(image_size[0])), np.uint8)
    dst*=255

    mask = np.zeros((int(image_size[1]), int(image_size[0])), np.uint8)
    mask = cv2.fillConvexPoly(mask, imagePoints.astype(np.int32), 255)

    dst[mask>0] = img[mask>0]

    # Metadata
    meta = {'n_markers': 1, 
        'm0_id': id_val,
        'm0_id_size': id_size,
        'm0_rvec_x': r_x,
        'm0_rvec_y': r_y,
        'm0_rvec_z': r_z,
        'm0_tvec_x': t_x,
        'm0_tvec_y': t_y,
        'm0_tvec_z': t_z,
        'm0_TL_x': int(imagePoints[0][0][0]),
        'm0_TL_y': int(imagePoints[0][0][1]),
        'm0_TR_x': int(imagePoints[1][0][0]),
        'm0_TR_y': int(imagePoints[1][0][1]),
        'm0_BR_x': int(imagePoints[2][0][0]),
        'm0_BR_y': int(imagePoints[2][0][1]),
        'm0_BL_x': int(imagePoints[3][0][0]),
        'm0_BL_y': int(imagePoints[3][0][1])}

    print("Writing image %s%i.tif..." % (folder, n_images))

    # Write tiff file as <number>.tif with attached metadata
    tifffile.imwrite(folder + '%i.tif'%n_images, dst, metadata=meta)

    # Increase number of images
    n_images+=1

print("Finished writing images, exiting...")