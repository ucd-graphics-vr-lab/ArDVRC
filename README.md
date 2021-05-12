

# ArDVRC

AR Directed Vector-Related Corners (ArDVRC) method of square fiducial marker detection and localization. This method is corner-based to help with issues of marker edge occlusions. This repository contains the ArDVRC static library and header file, sample applications, and data generation scripts.

### Example applications

##### Image example (`source/examples/image_example/`)

<img src="http://graphics.ucdenver.edu/img/ardvrc_image_example.png" alt="image_example" width="600" />

Processes individual images using ArDVRC and stitches them together.

##### Video example (`source/examples/video_example/`)

<img src="http://graphics.ucdenver.edu/img/ardvrc_video_example.gif" alt="image_example" width="600" />

Processes video from a file or a live camera stream using both ArDVRC and ArUco.

### System, libraries and tools

Tested on Windows 10 x64. macOS and Linux coming soon.

C++ projects in `source/examples/` require OpenCV built with contrib modules and CMake. Make sure to build OpenCV with contrib modules from source.

##### OpenCV + contrib

https://github.com/opencv/opencv

https://github.com/opencv/opencv_contrib

##### CMake

https://cmake.org/

For Python projects in `data/data_generation/`, please refer to the `README.md` in that directory.


### Building and running example applications

Run the following commands from inside the directory of the application you want to build:
```
mkdir build
cd build
cmake ..
cmake --build . --config "Release"
```

The .exe is now built and is in the `Release/` folder inside your build directory. There may be additional
files copied to that directory as well, such as a dictionary NPY file and sample images or videos. See
`README.md` files for each example application for more information on usage.

------------------------

Visit http://graphics.ucdenver.edu/ for more information on this project, and other AR/VR projects from
CU Denver's Computer Graphics and VR Lab.

<p align="center">
<img src="http://graphics.ucdenver.edu/img/cgvr_ucdenver.png" alt="image_example" width="500" />
</p>


