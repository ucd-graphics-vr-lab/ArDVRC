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

def get_error_bars(gt_percent_occlusion, method_frames):
    percentage_bar = np.arange(1, 101)

    gt_percent_occlusion_count = np.array([np.count_nonzero(gt_percent_occlusion==i) for i in percentage_bar])
    gt_occlusion_method = gt_percent_occlusion[method_frames]
    occlusion_counts = []

    ngroups = 20

    for i in range(ngroups):
        segment_start = i*(50000//ngroups)
        segment_end = (i+1)*(50000//ngroups)

        frames = method_frames[method_frames >= segment_start]
        frames = frames[frames < segment_end]
        total_percentages = gt_percent_occlusion[segment_start:segment_end]
        percentages = gt_percent_occlusion[frames]
        total_percent_occlusion_count = np.array([np.count_nonzero(total_percentages==i) for i in percentage_bar])
        percent_occlusion_count = np.array([np.count_nonzero(percentages==i) for i in percentage_bar])
        occlusion_counts.append((percent_occlusion_count/total_percent_occlusion_count)*100)
    '''segment_start = 0
    segment_end = 50000

    frames = method_frames[method_frames >= segment_start]
    frames = frames[frames < segment_end]
    total_percentages = gt_percent_occlusion[segment_start:segment_end]
    percentages = gt_percent_occlusion[frames]
    total_percent_occlusion_count = np.array([np.count_nonzero(total_percentages==i) for i in percentage_bar])
    percent_occlusion_count = np.array([np.count_nonzero(percentages==i) for i in percentage_bar])
    occlusion_counts.append((percent_occlusion_count/total_percent_occlusion_count)*100)'''
    occlusion_counts = np.array(occlusion_counts)

    avg = np.average(occlusion_counts, axis=0)
    std = np.std(occlusion_counts, axis=0)
    print(avg)
    print(std)

    return avg, std




gt_percent_occlusion = np.load('occlusion.npy')
frames_detected = np.load('frames.npy')

occlusion_identified = gt_percent_occlusion[frames_detected]

print(gt_percent_occlusion)
print(frames_detected)

# Custom data
'''ardvrc_frames = np.load(folder + 'results/ardvrc_frames.npy')
ardvrc_ids = np.load(folder +'results/ardvrc_ids.npy')
#ardvrc_corners = np.load(folder +'results/ardvrc_corners.npy')
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

# Get the ground truth occlusion percentage that align with ArUco frames 
gt_occlusion_aruco = gt_percent_occlusion[aruco_frames]

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


ardvrc_frames = ardvrc_frames[equivalent_ardvrc_ids]

ardvrc_nz = np.count_nonzero(equivalent_ardvrc_ids)
ardvrc_true_detections = ardvrc_nz
ardvrc_false_detections = ardvrc_ids.shape[0] - ardvrc_nz
ardvrc_no_detections = gt_ids.shape[0] - (ardvrc_true_detections + ardvrc_false_detections)

gt_frames_ardvrc = gt_frames[ardvrc_frames]
gt_ids_ardvrc = gt_ids[ardvrc_frames].reshape(-1)
gt_occlusion_ardvrc = gt_percent_occlusion[ardvrc_frames]

print('ArDVRC')
print('true detections:', ardvrc_true_detections)
print('false detections:', ardvrc_false_detections)
print('undetected:', ardvrc_no_detections)
print('time:', ardvrc_time_avg, '+/-', ardvrc_time_std, 'ms')
print()

# print("gt_occlusion_ardvrc: ", gt_occlusion_ardvrc)
# print("gt_occlusion_aruco: ", gt_occlusion_aruco)'''


#Create the percentage bar and count the occurrence of each percentage 
percentage_bar = np.arange(1, 101)

gt_percent_occlusion = np.ceil(gt_percent_occlusion)
gt_percent_occlusion_count = np.array([np.count_nonzero(gt_percent_occlusion==i) for i in percentage_bar])

#Ceil the occlusion to have integer values to be counted 
occlusion_identified = np.ceil(occlusion_identified)
#count the occurrence of each value of the percentage bar in ardvrc occlusion array
occlusion_count = np.array([np.count_nonzero(occlusion_identified==i) for i in percentage_bar])


occlusion_percentage = np.ceil((occlusion_count/gt_percent_occlusion_count) * 100 )
print(occlusion_percentage)


#Ceil the occlusion to have integer values to be counted 
'''aruco_occlusion = np.ceil(gt_occlusion_aruco)
#count the occurrence of each value of the percentage bar in aruco occlusion array
aruco_occlusion_count = np.array([np.count_nonzero(aruco_occlusion==i) for i in percentage_bar])

aruco_occlusion_percentage = np.ceil((aruco_occlusion_count/gt_percent_occlusion_count) * 100 )'''



ardvrc_avg, ardvrc_std = get_error_bars(gt_percent_occlusion, frames_detected)
#aruco_avg, aruco_std = get_error_bars(gt_percent_occlusion, aruco_frames)

print("gt_percent_occlusion_count", gt_percent_occlusion_count)
print("occlusion_count", occlusion_count)

print("occlusion_percentage", occlusion_percentage)

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)

# Create a figure
fig, ax = plt.subplots(figsize=(7, 4))

# N is the count in each bin, bins is the lower-limit of the bin
x_max = 35 #int(np.ceil(max(aruco_occlusion_count.max(), min(120,ardvrc_occlusion_count.max()))))
n_bins = 40 #np.arange(x_max) - 0.5


# print("x_max:", x_max)
# print("n_bins", n_bins)
# Display text box
props = dict(boxstyle='round', facecolor='cornsilk', alpha=0.5)

# What to print out in the text box in the upper right corner
#aruco_text = r"$\bf{ArUco Identification}$"#+"\nIdentified: %.2f%%" % (gt_frames_aruco.shape[0]/(gt_ids.shape[0]/100.0))
aruco_text = "ArUco Identification"
#ardvrc_text = r"$\bf{ArDVRC}$"#+"\nIdentified: %.2f%%" % (gt_frames_ardvrc.shape[0]/(gt_ids.shape[0]/100.0))

# Histograms of data
# N, bins, patches = ax.hist(aruco_occlusion, bins=n_bins, label=aruco_text, log=False, density=False, alpha=0.6 ) #color="#ffb26e"
# N, bins, patches = ax.hist(ardvrc_occlusion, bins=n_bins, label=ardvrc_text, log=False, density=False, alpha=0.6)
# print(N)
# print("aruco_occlusion_count: ", aruco_occlusion_count)

occlusion_percentage = np.insert(occlusion_percentage, 0, 100.0)
percentage_bar = np.insert(percentage_bar, 0, 0)

#ax.plot(percentage_bar[1:], ardvrc_avg, label=aruco_text, marker='o', markersize=2)

ax.fill_between(percentage_bar[1:], ardvrc_avg - ardvrc_std, ardvrc_avg + ardvrc_std, alpha=0.2)


ax.plot(percentage_bar, occlusion_percentage, label=aruco_text, marker='o', markersize=2)

# Limits, labels
ax.set_xlim(0, x_max+1)
ax.set_ylim(0,100)
ax.set_xlabel('Occlusion percentage (%)')
ax.set_ylabel('Identification percentage (%)')

ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Set the legend in the upper right corner
leg = ax.legend(loc='upper right')

# Center the legend marker at the top.
# <https://github.com/matplotlib/matplotlib/issues/12388#issuecomment-427539147>
hp = leg._legend_box.get_children()[1]
for vp in hp.get_children():
    for row in vp.get_children():
        row.align="bottom"

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles[::-1], labels[::-1], loc='upper right')

# Set the title to the stored dictionary value
#ax.set_title(data_dict[folder])
fig.tight_layout()

# Write PNG and PDF
#plt.savefig(folder + 'results/results_figure.png')
#plt.savefig(folder + 'results/results_figure.pdf')
plt.show()