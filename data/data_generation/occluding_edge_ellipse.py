import cv2
import numpy as np
import math
import random
import tifffile
from os import listdir
from os.path import isfile, join
from scipy.spatial.transform import Rotation as R

from skimage.draw import random_shapes

# Image size. Center point (cx, cy) for camera matrix taken from half the image size
image_size = (1280,720)

# Focal length info taken from ZED's website: 
# https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
focal_length = 700

scale = 1

remove_folder = True 

focal_length = int(focal_length*scale)
image_size = (int(image_size[0]*scale), int(image_size[1]*scale))

# Camera matrix (imitating ZED camera)
zed_camera_matrix = np.array([[focal_length, 0.0, image_size[0]/2],[0.0, focal_length, image_size[1]/2],[0.0, 0.0, 1.0]])

# Source data directory
source_dir = 'datasets/control/'
dest_dir = 'datasets/occluding_edge_ellipse/'

print("\n*** ArDVRC EDGE OCCLUSION GENERATOR ***\n")
print("This script takes a dataset and adds an ellipse to occlude an edge of the marker.\n")
print("The input dataset is %s. Make sure this is correct!\n" % source_dir)
print("WARNING: THE GENERATED FILES TAKE UP A LOT OF HARDDRIVE SPACE. Make sure your computer has enough space!\n")
print("WARNING: This script tends to take longer than the others due to the randomization of the shape location. This will be fixed in future development.\n")

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
import shutil

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
else:
    if remove_folder: 
        shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
    else:
        print('ERROR: The output folder %s already exists. Move, rename or delete this directory and run this program again.' % dest_dir)
        exit()

filenames = [f for f in listdir(source_dir) if (isfile(join(source_dir, f)) and f.endswith('.tif'))]

def get_rvec_tvec_from_metadata(meta):
    rvec = np.array([meta['m0_rvec_x'], meta['m0_rvec_y'], meta['m0_rvec_z']])
    tvec = np.array([meta['m0_tvec_x'], meta['m0_tvec_y'], meta['m0_tvec_z']])
    return rvec, tvec

def get_marker_corners_from_metadata(meta):
    TL = (meta['m0_TL_x'], meta['m0_TL_y'])
    TR = (meta['m0_TR_x'], meta['m0_TR_y'])
    BR = (meta['m0_BR_x'], meta['m0_BR_y'])
    BL = (meta['m0_BL_x'], meta['m0_BL_y'])
    points = np.array([TL,TR,BR,BL])
    return points

def get_id_info_from_metadata(meta):
    id_val = meta['m0_id']
    id_size = meta['m0_id_size']
    return id_val, id_size


print("Writing images...")
# Source folder
for f in filenames:
    with tifffile.TiffFile(join(source_dir, f)) as tif:
        a = tif.asarray()

        meta = tif.shaped_metadata[0]
        meta.pop('shape',None)

        points = get_marker_corners_from_metadata(meta)

        rvec, tvec = get_rvec_tvec_from_metadata(meta)

        id_val, id_size = get_id_info_from_metadata(meta)

        # 2D plane in 3D space (depth==0)
        plane_points = np.array([[int(-id_size/2),int(-id_size/2),0], [int(id_size/2),int(-id_size/2),0], [int(id_size/2),int(id_size/2),0], [int(-id_size/2),int(id_size/2),0]], dtype=np.float32)

        outer_plane_points = np.array([[int((-id_size*2)/3),int((-id_size*2)/3),0], [int((id_size*2)/3),int((-id_size*2)/3),0], [int((id_size*2)/3),int((id_size*2)/3),0], [int((-id_size*2)/3),int((id_size*2)/3),0]], dtype=np.float32)

        # Convert from Euler rotations to Axis-Angle (Rodrigues) vector
        rvec = R.from_euler('XYZ', np.array(rvec), degrees=True).as_rotvec()

        tvec = tvec.reshape(3,1).astype(np.float32)
        imagePoints, j = cv2.projectPoints(plane_points, rvec, tvec, zed_camera_matrix, np.zeros((5,1),np.float32))
        image_points_outer, j = cv2.projectPoints(outer_plane_points, rvec, tvec, zed_camera_matrix, np.zeros((5,1),np.float32))

        # Set to true when we have overlap between the shapes and the marker
        overlap = False
        while(not overlap):

            a_color = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

            shape_mask = np.zeros_like(a)

            # Get a random shapes image
            result, _ = random_shapes(a.shape, min_shapes=1, max_shapes=1, min_size=20, max_size=400, allow_overlap=False, shape='ellipse')#, multichannel=False)#intensity_range=((0,255),))
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            marker_mask = np.zeros_like(a)
            marker_mask = cv2.fillConvexPoly(marker_mask, imagePoints.astype(np.int32), 255)
            
            #count non-zero pixels to calculate the area of the marker 
            marker_mask_non_zero_pixels =  np.count_nonzero(marker_mask)
            

            shape_mask[result_gray<255] = 255     
            
            overlap_pixels = cv2.bitwise_and(shape_mask, marker_mask)

            #count non-zero pixels to calculate the occluded are 
            overlap_pixels_non_zero = np.count_nonzero(overlap_pixels)
            

            #calculate the percentage of the occluded area by diving overlapped pixels(mask and object)
            occlusion_percentage = (overlap_pixels_non_zero/marker_mask_non_zero_pixels) * 100

            # print("occlusion_percentage: ",occlusion_percentage)             

            combined_pixels = cv2.bitwise_or(shape_mask, marker_mask)

            if overlap_pixels.sum() > 0:
                if combined_pixels.sum() == marker_mask.sum() or combined_pixels.sum() == shape_mask.sum():
                    continue
                else:
                    overlap = True
                    #add occlusion percentage to metadata
                    meta["occlusion_percentage"] = round(occlusion_percentage, 3)
                    

            a_color[shape_mask>0] = result[shape_mask>0]
          
        tifffile.imwrite(join(dest_dir,f), a_color, metadata=meta)

print("Finished writing images, exiting...")