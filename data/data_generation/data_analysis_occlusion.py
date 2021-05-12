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

# Folders and titles
data_dict = {
    'datasets/control/': 'Control data',
    'datasets/occluding_edge_ellipse/': 'Edge occlusions data (percentage)',
    'datasets/background_shapes/': 'Background shapes data',
    'datasets/decreased_lighting/': 'Lighting data',
    'datasets/motion_blur_horizontal_kernel_10/': 'Motion blur data (K=10)',
    'datasets/motion_blur_horizontal_kernel_20/': 'Motion blur data (K=20)',
    'datasets/motion_blur_horizontal_kernel_30/': 'Motion blur data (K=30)',
    'datasets/noise/': 'Noise data'}

# Data folder (uncomment the one you want, it should have a folder called 'results' in it with some npy files)
# folder = 'datasets/control/'
folder = 'datasets/occluding_edge_ellipse/'
#folder = 'datasets/background_shapes/'
#folder = 'datasets/decreased_lighting/'
#folder = 'datasets/motion_blur_horizontal_kernel_10/'
#folder = 'datasets/motion_blur_horizontal_kernel_20/'
#folder = 'datasets/motion_blur_horizontal_kernel_30/'
#folder = 'datasets/noise/'

# Make sure the results folder actually exists
results_path = Path(folder + 'results/')
if not results_path.is_dir():
    print('Results directory for this dataset does not exist, please run ardvrc_aruco_comparison.cpp first.')
    exit()

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



gtd = get_ground_truth_data(folder)

# Need to make sure we account for a reduced-size image
gt_corners = gtd.corners
gt_corners = gt_corners/2
gt_ids = gtd.ids
gt_frames = gtd.frames
gt_percent_occlusion = gtd.percent_occlusion


identification_only_frames = np.load('frames.npy')
identification_only_occlusion = gt_percent_occlusion[identification_only_frames]



# Custom data
ardvrc_frames = np.load(folder + 'results/ardvrc_frames.npy')
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

'''for idx, e in enumerate(equivalent_ardvrc_ids):
    if e == 1:
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+1])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+2])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+3])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+4])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+5])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+6])
        ardvrc_corners_tmp.append(ardvrc_corners[idx*8+7])

ardvrc_corners = np.array(ardvrc_corners_tmp)'''
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
# print("gt_occlusion_aruco: ", gt_occlusion_aruco)


#Create the percentage bar and count the occurrence of each percentage 
percentage_bar = np.arange(1, 101)

gt_percent_occlusion = np.ceil(gt_percent_occlusion)
gt_percent_occlusion_count = np.array([np.count_nonzero(gt_percent_occlusion==i) for i in percentage_bar])

#Ceil the occlusion to have integer values to be counted 
ardvrc_occlusion = np.ceil(gt_occlusion_ardvrc)
#count the occurrence of each value of the percentage bar in ardvrc occlusion array
ardvrc_occlusion_count = np.array([np.count_nonzero(ardvrc_occlusion==i) for i in percentage_bar])

ardvrc_occlusion_percentage = np.ceil((ardvrc_occlusion_count/gt_percent_occlusion_count) * 100 )


#Ceil the occlusion to have integer values to be counted 
aruco_occlusion = np.ceil(gt_occlusion_aruco)
#count the occurrence of each value of the percentage bar in aruco occlusion array
aruco_occlusion_count = np.array([np.count_nonzero(aruco_occlusion==i) for i in percentage_bar])

aruco_occlusion_percentage = np.ceil((aruco_occlusion_count/gt_percent_occlusion_count) * 100 )


#Ceil the occlusion to have integer values to be counted 
identification_only_occlusion = np.ceil(identification_only_occlusion)
#count the occurrence of each value of the percentage bar in ardvrc occlusion array
identification_only_occlusion_count = np.array([np.count_nonzero(identification_only_occlusion==i) for i in percentage_bar])

identification_only_occlusion_percentage = np.ceil((identification_only_occlusion_count/gt_percent_occlusion_count) * 100 )



ardvrc_avg, ardvrc_std = get_error_bars(gt_percent_occlusion, ardvrc_frames)
aruco_avg, aruco_std = get_error_bars(gt_percent_occlusion, aruco_frames)
identification_only_avg, identification_only_std = get_error_bars(gt_percent_occlusion, identification_only_frames)

print("gt_percent_occlusion_count", gt_percent_occlusion_count)
print("ardvrc_occlusion_count", ardvrc_occlusion_count)
print("aruco_occlusion_count", aruco_occlusion_count)

print("ardvrc_occlusion_percentage", ardvrc_occlusion_percentage)
print("aruco_occlusion_percentage", aruco_occlusion_percentage)

# print("gt_occlusion_ardvrc:", gt_occlusion_ardvrc)


# print("occlusion_ardvrc: ", percentage_bar)
'''
# Get ground truth corners that correspond with the found ardvrc frames
gt_corners_ardvrc = gt_corners[ardvrc_frames]

# Reshape data
gt_corners_aruco = gt_corners_aruco.reshape(-1, 2)
gt_corners_ardvrc = gt_corners_ardvrc.reshape(-1, 2)
aruco_corners_gt = aruco_corners.reshape(-1, 2)
ardvrc_corners_gt = ardvrc_corners.reshape(-1, 2)

# print("gt_corners_aruco: ", gt_corners_aruco)

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
'''

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
aruco_text = r"$\bf{ArUco}$"#+"\nIdentified: %.2f%%" % (gt_frames_aruco.shape[0]/(gt_ids.shape[0]/100.0))
ardvrc_text = r"$\bf{ArDVRC}$"#+"\nIdentified: %.2f%%" % (gt_frames_ardvrc.shape[0]/(gt_ids.shape[0]/100.0))
idonly_text = r"$\bf{Identification \; only}$"

# Histograms of data
# N, bins, patches = ax.hist(aruco_occlusion, bins=n_bins, label=aruco_text, log=False, density=False, alpha=0.6 ) #color="#ffb26e"
# N, bins, patches = ax.hist(ardvrc_occlusion, bins=n_bins, label=ardvrc_text, log=False, density=False, alpha=0.6)
# print(N)
# print("aruco_occlusion_count: ", aruco_occlusion_count)

aruco_occlusion_percentage = np.insert(aruco_occlusion_percentage, 0, 99.31)
ardvrc_occlusion_percentage = np.insert(ardvrc_occlusion_percentage, 0, 98.61)
identification_only_occlusion_percentage = np.insert(identification_only_occlusion_percentage, 0, 99.972)
percentage_bar = np.insert(percentage_bar, 0, 0)


'''ax.plot(percentage_bar[1:], aruco_avg, label=aruco_text, marker='o', markersize=2)
ax.plot(percentage_bar[1:], ardvrc_avg, label=ardvrc_text, marker='o', markersize=2)

ax.fill_between(percentage_bar[1:], aruco_avg - aruco_std, aruco_avg + aruco_std, alpha=0.2)
ax.fill_between(percentage_bar[1:], ardvrc_avg - ardvrc_std, ardvrc_avg + ardvrc_std, alpha=0.2)'''

ax.plot(percentage_bar, aruco_occlusion_percentage, label=aruco_text, marker='o', markersize=2)
ax.plot(percentage_bar, ardvrc_occlusion_percentage, label=ardvrc_text, marker='o', markersize=2)
ax.plot(percentage_bar, identification_only_occlusion_percentage, label=idonly_text, marker='o', markersize=2)

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
plt.savefig(folder + 'results/results_figure.png')
plt.savefig(folder + 'results/results_figure.pdf')
plt.show()