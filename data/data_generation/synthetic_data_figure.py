#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tifffile

folders = ['random_control_10000/',  # Didn't improve # of detections, but did improve corners
           'random_occluding_edge_ellipse_10000/',
           'random_noise_10000/',
           
           'random_shapes_10000/',
           'random_horizontal_motion_blur/kernel_30/',
           'random_lighting_10000/decreased_lighting_50%/',
           
           #'random_occluding_edge_shape/',
           #'random_vertical_motion_blur/kernel_30/',
           
           #'random_foreground_shapes/',
           ]

labels = ['Control', 'Edge occlusion', 'Noise', 'Background shapes', 'Motion blur', 'Lighting']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))

n = 653

for idx,f in enumerate(folders):
    # Open the TIFF file
    with tifffile.TiffFile('D:/research/ardvrc/data_generation/custom_DICT_4X4_64_ENTROPY_1000/' + f + str(n)+'.tif') as tif:
        frame = tif.asarray()
        meta = tif.shaped_metadata

    if frame is None:
        exit()

    if len(frame.shape) == 3:
        axes[int(idx/2), idx%2].imshow(frame)#, origin='lower', cmap='Greys_r')
    else:
        axes[int(idx/2), idx%2].imshow(frame,cmap='gray', vmin=0, vmax=255)
    axes[int(idx/2), idx%2].xaxis.set_ticks([])
    axes[int(idx/2), idx%2].yaxis.set_ticks([])
    axes[int(idx/2), idx%2].set_xlabel(labels[idx], fontsize=14)
fig.tight_layout(pad=1)
plt.savefig('synthetic_data.pdf')
plt.savefig('synthetic_data.png')
plt.show()