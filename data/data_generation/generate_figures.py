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
from data_analysis import DataCollector, Metadata, get_ground_truth_data, save_ground_truth_data

# Folders and titles
data_dict = {
    'datasets/control/': 'Control data',
    'datasets/occluding_edge_ellipse/': 'Edge occlusions data',
    'datasets/noise/': 'Noise data',
    'datasets/background_shapes/': 'Background shapes data',
    #'datasets/motion_blur_horizontal_kernel_10/': 'Motion blur data (K=10)',
    'datasets/motion_blur_horizontal_kernel_20/': 'Motion blur data (K=20)',
    #'datasets/motion_blur_horizontal_kernel_30/': 'Motion blur data (K=30)',
    'datasets/decreased_lighting/': 'Lighting data'
    }

# Create a figure
fig, ax = plt.subplots(3, 2, figsize=(10, 8))#, sharex=True, sharey=True)

# Display text box
props = dict(boxstyle='round', facecolor='cornsilk', alpha=0.5)

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for idxv, folder in enumerate(data_dict.keys()):

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

    # N is the count in each bin, bins is the lower-limit of the bin
    x_max = 120#int(np.ceil(max(gt_aruco_distances.max(), min(120,gt_ardvrc_distances.max()))))
    n_bins = np.arange(x_max+2) - 0.5

    # What to print out in the text box in the upper right corner
    aruco_text = r"$\bf{ArUco}$"+"\nIdentified: %.2f%%\nError: %.2f \u00B1 %.2f px" % (gt_frames_aruco.shape[0]/(gt_ids.shape[0]/100.0),gt_aruco_distances_avg,gt_aruco_distances_std)
    ardvrc_text = r"$\bf{ArDVRC}$"+"\nIdentified: %.2f%%\nError: %.2f \u00B1 %.2f px" % (gt_frames_ardvrc.shape[0]/(gt_ids.shape[0]/100.0),gt_ardvrc_distances_avg,gt_ardvrc_distances_std)

    row = int(math.floor(idxv/2))
    col = int(idxv%2)

    # Histograms of data
    N, bins, patches = ax[row, col].hist(gt_aruco_distances, bins=n_bins, label=aruco_text, log=True, density=False, alpha=0.6)
    N, bins, patches = ax[row, col].hist(gt_ardvrc_distances, bins=n_bins, label=ardvrc_text, log=True, density=False, alpha=0.6)

    # Limits, labels
    #ax[row, col].set_xlim(-1.00001, x_max+1)
    ax[row, col].set_xlim([-1.00001, x_max+1])
    ax[row, col].set_ylim([0, pow(10,4.6)])
    ax[row, col].set_yscale('log')
    ax[row, col].set_xlabel('Corner position error (px)')
    ax[row, col].set_ylabel('Corner counts')
    ax[row, col].xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Set the legend in the upper right corner
    #leg = ax[row, col].legend(loc='upper right')

    handles, labels = ax[row,col].get_legend_handles_labels()
    leg = ax[row, col].legend(handles[::-1], labels[::-1], loc='upper right')

    # Center the legend marker at the top.
    # <https://github.com/matplotlib/matplotlib/issues/12388#issuecomment-427539147>
    hp = leg._legend_box.get_children()[1]
    for vp in hp.get_children():
        for row_vp in vp.get_children():
            row_vp.align="bottom"

    # Set the title to the stored dictionary value
    ax[row, col].set_title(data_dict[folder])
    
    ax[row, col].text(-0.1, 1.1, letters[idxv], transform=ax[row, col].transAxes,
      fontsize=12, va='top')

    
fig.tight_layout()
# Write PNG and PDF
plt.savefig(folder + 'results/results_figure.png')
plt.savefig(folder + 'results/results_figure.pdf')
plt.show()
