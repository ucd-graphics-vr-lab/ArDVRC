

# ArDVRC

AR Directed Vector-Related Corners (ArDVRC) method of square fiducial marker detection and localization. This method is corner-based to help with issues of marker edge occlusions. This repository contains sample applications and data generation scripts.

<img src="http://graphics.ucdenver.edu/img/ardvrc_image_example.png" alt="image_example" style="zoom: 50%;" />

<img src="http://graphics.ucdenver.edu/img/ardvrc_video_example.gif" alt="image_example" style="zoom: 100%;" />

### System

Tested on Windows 10 x64. Will soon test on OSX and Linux.

### Required libraries and tools

C++ projects in `source/examples/` require OpenCV built with contrib modules and CMake. OpenCV with contrib modules will need to be built from source.

##### OpenCV + contrib

https://github.com/opencv/opencv

https://github.com/opencv/opencv_contrib

##### CMake

https://cmake.org/

For Python projects in `data/data_generation/`, please refer to the `README.md` in that directory.

### Building example applications

Run the following commands from inside the directory of the application you want to build:
```
mkdir build
cd build
cmake ..
cmake --build . --config "Release"
```

<img src="http://graphics.ucdenver.edu/img/cgvr_lab_black.png" alt="image_example" style="zoom: 2%;" />

<img src="http://graphics.ucdenver.edu/img/cu_denver_logo.jpg" alt="image_example" style="zoom: 40%;" />

