# Data analysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
import tifffile
from scipy.spatial.transform import Rotation as R
import glob
import os.path
from pathlib import Path


class DataCollector(object):
    ''' Stores and manages results data collected for data analysis. '''

    def __init__(self):
        self.frames = None
        self.ids = None
        self.corners = None
        self.percent_occlusion = None

    def add_data(self, frame_number, id_val, corners, occlusion=None):
        if type(id_val)==int:
            id_val = np.array([id_val])
        corners = corners.reshape(4, 2)
        id_val = id_val.reshape(-1)

        if self.frames is None:
            self.frames = np.array([frame_number], dtype=np.int32)
        else:
            self.frames = np.concatenate((self.frames, [frame_number]))
        if self.ids is None:
            self.ids = np.array([id_val], dtype=np.int32)
        else:
            self.ids = np.concatenate((self.ids, [id_val]))
        if self.corners is None:
            self.corners = np.array([corners.astype(np.int32)], dtype=np.int32)
        else:
            self.corners = np.concatenate((self.corners, [corners]))

        # Only add occlusion data if it's available
        if occlusion is not None:
            if self.percent_occlusion is None:
                self.percent_occlusion = np.array([occlusion], dtype=np.float32)
            else:
                self.percent_occlusion = np.concatenate((self.percent_occlusion, [occlusion]))


class Metadata(object):
    ''' Parses metadata stored within the image (TIF) file '''

    def __init__(self, meta):
        self.parse_metadata(meta)

    def parse_metadata(self, meta):
        self._meta = meta
        self.n_markers = meta[0]['n_markers']
        self.ids = np.zeros((self.n_markers),dtype=np.int32)

        self.id_sizes = np.zeros((self.n_markers), dtype=np.int32)
        self.rvecs = np.zeros((self.n_markers, 3), dtype=np.float64)
        self.tvecs = np.zeros((self.n_markers, 3), dtype=np.int32)

        self.corners = np.zeros((self.n_markers, 4, 2), dtype=np.int32)

        # This is only available for certain datasets
        self.percent_occlusion = None

        for i in range(self.n_markers):
            prefix = 'm'+str(i)+'_'
            #print(meta[0][prefix+'id'])
            self.ids[i] = meta[0][prefix+'id']
            self.id_sizes[i] = meta[0][prefix+'id_size']
            self.rvecs[i] = np.array([meta[0][prefix+'rvec_x'], meta[0][prefix+'rvec_y'], meta[0][prefix+'rvec_z']], dtype=np.float64)

            # Convert from Euler rotations to Axis-Angle (Rodrigues) vector
            self.rvecs[i] = R.from_euler('XYZ', self.rvecs[i], degrees=True).as_rotvec()
            self.tvecs[i] = np.array([meta[0][prefix+'tvec_x'], meta[0][prefix+'tvec_y'], meta[0][prefix+'tvec_z']])

            self.corners[i][0][0] = meta[0][prefix+'TL_x'] # Top left x/y values
            self.corners[i][0][1] = meta[0][prefix+'TL_y']
            self.corners[i][1][0] = meta[0][prefix+'TR_x'] # Top right x/y values
            self.corners[i][1][1] = meta[0][prefix+'TR_y']
            self.corners[i][2][0] = meta[0][prefix+'BR_x'] # Bottom right x/y values
            self.corners[i][2][1] = meta[0][prefix+'BR_y']
            self.corners[i][3][0] = meta[0][prefix+'BL_x'] # Bottom left x/y values
            self.corners[i][3][1] = meta[0][prefix+'BL_y']
            if 'occlusion_percentage' in meta[0]:
                self.percent_occlusion = meta[0]['occlusion_percentage']

def get_ground_truth_data(input_folder): 

    # Folder containing all of the tif files
    ground_truth_data = DataCollector()

    # If we've already read the data from the files and stored them as NPY files, load those
    if os.path.isfile(input_folder + 'results/ground_truth_frames.npy'):
        print("Found saved ground truth data, reading...")
        ground_truth_data.frames = np.load(input_folder + 'results/ground_truth_frames.npy')
        ground_truth_data.ids = np.load(input_folder + 'results/ground_truth_ids.npy')
        ground_truth_data.corners = np.load(input_folder + 'results/ground_truth_corners.npy')
        if input_folder == 'datasets/occluding_edge_ellipse/':
            ground_truth_data.percent_occlusion = np.load(input_folder + 'results/ground_truth_percent_occlusion.npy')
        print("Done reading ground truth data.\n")

    # Read each individual frame and add the metadata from each
    else:
        print("Could not find ground truth data, generating...")
        # Get the number of files in the folder ending with .tif
        nfiles = len(glob.glob1(input_folder,"*.tif"))

        if nfiles == 0:
            print("No TIFF files exist for this dataset, exiting...")
            exit()

        # Get the metadata for each file
        for n in range(nfiles):
            with tifffile.TiffFile(input_folder + str(n)+'.tif') as tif:
                meta = tif.shaped_metadata
                # Parse the data, store as a class representation
                data = Metadata(meta)
                ground_truth_data.add_data(n, data.ids.reshape(-1), data.corners, data.percent_occlusion)

        save_ground_truth_data(ground_truth_data, input_folder)
        print("Done generating ground truth data.\n")

    return ground_truth_data

def save_ground_truth_data(ground_truth, output_folder):
    np.save(output_folder + 'results/ground_truth_frames.npy', ground_truth.frames)
    np.save(output_folder + 'results/ground_truth_ids.npy', ground_truth.ids)
    np.save(output_folder + 'results/ground_truth_corners.npy', ground_truth.corners)
    if ground_truth.percent_occlusion is not None:
        np.save(output_folder + 'results/ground_truth_percent_occlusion.npy', ground_truth.percent_occlusion)

if __name__=='__main__':
    # Folders and titles
    data_dict = {
        'datasets/control/': 'Control data',
        'datasets/occluding_edge_ellipse/': 'Edge occlusions data',
        'datasets/background_shapes/': 'Background shapes data',
        'datasets/decreased_lighting/': 'Lighting data',
        'datasets/motion_blur_horizontal_kernel_10/': 'Motion blur data (K=10)',
        'datasets/motion_blur_horizontal_kernel_20/': 'Motion blur data (K=20)',
        'datasets/motion_blur_horizontal_kernel_30/': 'Motion blur data (K=30)',
        'datasets/noise/': 'Noise data'}

    # Data folder (uncomment the one you want, it should have a folder called 'results' in it with some npy files)
    #folder = 'datasets/control/'
    #folder = 'datasets/occluding_edge_ellipse/'
    #folder = 'datasets/background_shapes/'
    #folder = 'datasets/decreased_lighting/'
    #folder = 'datasets/motion_blur_horizontal_kernel_10/'
    folder = 'datasets/motion_blur_horizontal_kernel_20/'
    #folder = 'datasets/motion_blur_horizontal_kernel_30/'
    #folder = 'datasets/noise/'

    # Make sure the results folder actually exists
    results_path = Path(folder + 'results/')
    if not results_path.is_dir():
        print('Results directory for this dataset does not exist, please run ardvrc_aruco_comparison.cpp first.')
        exit()

    gtd = get_ground_truth_data(folder)

    # Need to make sure we account for a reduced-size image
    gt_corners = gtd.corners
    gt_corners = gt_corners/2
    gt_ids = gtd.ids
    gt_frames = gtd.frames

    # Custom data
    ardvrc_frames = np.load(folder + 'results/ardvrc_frames.npy')
    ardvrc_ids = np.load(folder +'results/ardvrc_ids.npy')
    ardvrc_corners = np.load(folder +'results/ardvrc_corners.npy')
    ardvrc_times = np.load(folder + 'results/ardvrc_time.npy')

    # ArUco data (frames, ids, corners, time)
    aruco_frames = np.load(folder +'results/aruco_frames.npy')
    aruco_ids = np.load(folder +'results/aruco_ids.npy')
    aruco_corners = np.load(folder +'results/aruco_corners.npy')
    aruco_times = np.load(folder + 'results/aruco_time.npy')

    # Time averages and standard deviations
    ardvrc_time_avg = np.average(ardvrc_times)
    ardvrc_time_std = np.std(ardvrc_times)

    # Get the ground truth frames that align with the found standard frames
    gt_frames_aruco = gt_frames[aruco_frames]

    # Get the ground truth IDs that align with the ArUco frames
    gt_ids_aruco = gt_ids[aruco_frames].reshape(-1)
    equivalent_aruco_ids = gt_ids_aruco == aruco_ids

    # Calculate the number of true/false detections for ArUco
    aruco_nz = np.count_nonzero(equivalent_aruco_ids)
    aruco_true_detections = aruco_nz
    aruco_false_detections = aruco_ids.shape[0] - aruco_nz
    aruco_no_detections = gt_ids.shape[0] - (aruco_true_detections + aruco_false_detections)

    # Print info about ArUco detections
    print('ArUco')
    print('true detections:', aruco_true_detections)
    print('false detections:', aruco_false_detections)
    print('undetected:', aruco_no_detections)
    print()

    # Get the ground truth corners that align with the standard frames
    gt_corners_aruco = gt_corners[aruco_frames]

    # Align ground truth with ArDVRC detections
    gt_ids_ardvrc = gt_ids[ardvrc_frames].reshape(-1)
    equivalent_ardvrc_ids = gt_ids_ardvrc == ardvrc_ids
    ardvrc_ids = ardvrc_ids[equivalent_ardvrc_ids]
    ardvrc_corners_tmp = []

    for idx, e in enumerate(equivalent_ardvrc_ids):
        if e == 1:
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+1])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+2])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+3])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+4])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+5])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+6])
            ardvrc_corners_tmp.append(ardvrc_corners[idx*8+7])

    ardvrc_corners = np.array(ardvrc_corners_tmp)
    ardvrc_frames = ardvrc_frames[equivalent_ardvrc_ids]

    ardvrc_nz = np.count_nonzero(equivalent_ardvrc_ids)
    ardvrc_true_detections = ardvrc_nz
    ardvrc_false_detections = ardvrc_ids.shape[0] - ardvrc_nz
    ardvrc_no_detections = gt_ids.shape[0] - (ardvrc_true_detections + ardvrc_false_detections)

    gt_frames_ardvrc = gt_frames[ardvrc_frames]
    gt_ids_ardvrc = gt_ids[ardvrc_frames].reshape(-1)

    print('ArDVRC')
    print('true detections:', ardvrc_true_detections)
    print('false detections:', ardvrc_false_detections)
    print('undetected:', ardvrc_no_detections)
    print('time:', ardvrc_time_avg, '+/-', ardvrc_time_std, 'ms')
    print()

    # Get ground truth corners that correspond with the found ardvrc frames
    gt_corners_ardvrc = gt_corners[ardvrc_frames]

    # Reshape data
    gt_corners_aruco = gt_corners_aruco.reshape(-1, 2)
    gt_corners_ardvrc = gt_corners_ardvrc.reshape(-1, 2)
    aruco_corners_gt = aruco_corners.reshape(-1, 2)
    ardvrc_corners_gt = ardvrc_corners.reshape(-1, 2)

    # Get distances between aruco corners and the ground truth and ardvrc corners and the ground truth
    gt_aruco_distances = np.array([math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ) for p1, p2 in zip(gt_corners_aruco, aruco_corners_gt)])
    gt_ardvrc_distances = np.array([math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ) for p1, p2 in zip(gt_corners_ardvrc, ardvrc_corners_gt)])

    a = np.where(gt_ardvrc_distances > 100)[0]//4
    print(a)
    print(ardvrc_frames[a])

    # Statistics for the distances
    gt_aruco_distances_avg = np.average(gt_aruco_distances)
    gt_ardvrc_distances_avg = np.average(gt_ardvrc_distances)
    gt_aruco_distances_std = np.std(gt_aruco_distances)
    gt_ardvrc_distances_std = np.std(gt_ardvrc_distances)

    # ArUco and ArDVRC statistics
    print('ArUco detections', gt_frames_aruco.shape[0])
    print('ArDVRC detections', gt_frames_ardvrc.shape[0])
    print('ArUco avg', gt_aruco_distances_avg)
    print('ArUco std', gt_aruco_distances_std)
    print('ArDVRC avg', gt_ardvrc_distances_avg)
    print('ArDVRC std', gt_ardvrc_distances_std)

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # N is the count in each bin, bins is the lower-limit of the bin
    x_max = int(np.ceil(max(gt_aruco_distances.max(), min(120,gt_ardvrc_distances.max()))))
    n_bins = np.arange(x_max+2) - 0.5

    # Display text box
    props = dict(boxstyle='round', facecolor='cornsilk', alpha=0.5)

    # What to print out in the text box in the upper right corner
    aruco_text = r"$\bf{ArUco}$"+"\nIdentified: %.2f%%\nError: %.2f \u00B1 %.2f px" % (gt_frames_aruco.shape[0]/(gt_ids.shape[0]/100.0),gt_aruco_distances_avg,gt_aruco_distances_std)
    ardvrc_text = r"$\bf{ArDVRC}$"+"\nIdentified: %.2f%%\nError: %.2f \u00B1 %.2f px" % (gt_frames_ardvrc.shape[0]/(gt_ids.shape[0]/100.0),gt_ardvrc_distances_avg,gt_ardvrc_distances_std)

    # Histograms of data
    N, bins, patches = ax.hist(gt_aruco_distances, bins=n_bins, label=aruco_text, log=True, density=False, alpha=0.6)
    N, bins, patches = ax.hist(gt_ardvrc_distances, bins=n_bins, label=ardvrc_text, log=True, density=False, alpha=0.6)

    # Limits, labels
    ax.set_xlim(-1.00001, x_max+1)
    ax.set_xlabel('Corner position error (px)')
    ax.set_ylabel('Corner counts')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Set the legend in the upper right corner
    leg = ax.legend(loc='upper right')

    # Center the legend marker at the top.
    # <https://github.com/matplotlib/matplotlib/issues/12388#issuecomment-427539147>
    hp = leg._legend_box.get_children()[1]
    for vp in hp.get_children():
        for row in vp.get_children():
            row.align="bottom"

    # Set the title to the stored dictionary value
    ax.set_title(data_dict[folder])
    fig.tight_layout()

    # Write PNG and PDF
    plt.savefig(folder + 'results/results_figure.png')
    plt.savefig(folder + 'results/results_figure.pdf')
    plt.show()
