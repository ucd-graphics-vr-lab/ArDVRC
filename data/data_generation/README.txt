ArDVRC Dataset Generation Scripts

This folder contains all of the Python scripts used to generate the synthetic dataset from the paper.
Run these scripts to generate your own!

PYTHON VERSION: 3.x
REQUIRED PYTHON MODULES (install with pip): 
	opencv-contrib-python, opencv-python, numpy, tifffile, scipy, scikit-image
TESTED ON: Windows (although this should be cross-platform in theory, will confirm later)

USAGE:
	1) Generate the control dataset first. Run:

		python data_generator.py
	   
	   You will be prompted for the number of images you want to generate, and asked if you want to continue.
	   WARNING: Choosing a high number of images to generate will eat up your harddrive space. USE WITH CAUTION.
 
	   Generated images are placed in datasets/control/. If you look at the image tags you will notice some
	   metadata about the ID of the marker, the pose and marker size, corners of the marker, etc.

	2) Run other scripts on the generated control data to create different image effects.
	   All results will be placed in the datasets/ folder. The available scripts are:
		
		background_shapes.py - Adds colorful background shapes to the image.
		decreased_lighting.py - Synthesizes reduced lighting by reducing contrast.
		motion_blur.py - Adds motion blur. You choose the amount and direction via prompts.
 		noise.py - Adds salt-and-pepper noise in the foreground.
		occluding_edge_ellipse.py - Adds a randomly sized/colored/positioned ellipse over an edge of the marker.
		    This is to simulate finger occlusions.

	3) You can modify the source directory for any of the scripts to add multiple effects. For example,
	   running noise.py on a dataset you created using motion_blur.py will created a blurred and noisy image.

