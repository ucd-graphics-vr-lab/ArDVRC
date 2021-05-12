

# ArDVRC

AR Directed Vector-Related Corners (ArDVRC) method of square fiducial marker detection and localization. This method is corner-based to help with issues of marker edge occlusions. This repository contains the ArDVRC static library and header file, sample applications, and data generation scripts.

### Example applications

#### Image example (`source/examples/image_example/`)

<img src="http://graphics.ucdenver.edu/img/ardvrc_image_example.png" alt="image_example" width="600" />

Processes individual images using ArDVRC and stitches them together.

#### Video example (`source/examples/video_example/`)

<img src="http://graphics.ucdenver.edu/img/ardvrc_video_example.gif" alt="image_example" width="600" />

Processes video from a file or a live camera stream using both ArDVRC and ArUco.

### System, libraries and tools

ArDVRC has been tested in Windows 10 (x64). macOS and Linux coming soon.

The provided example applications are built using CMake, and require OpenCV to be installed with contrib modules. You will need to install OpenCV from source to include the contrib modules. If your system already has CMake and OpenCV with the contrib modules installed, you can skip to the next section. 

To set up your environment, follow these steps:

#### 1) Install CMake

Go to [CMake's Download page](https://cmake.org/download/) and download the appropriate installer. Run the installer, ensuring that you choose to **add CMake to the system path**.

#### 2) Install OpenCV with contrib modules

Download or clone [OpenCV](https://github.com/opencv/opencv) and the [OpenCV contrib modules](https://github.com/opencv/opencv_contrib) and extract/place them in the directory of your choice. Follow [this tutorial](https://cv-tricks.com/how-to/installation-of-opencv-4-1-0-in-windows-10-from-source/) to build OpenCV with contrib modules using CMake. Additionally, you will need to **make sure your OpenCV installation's `bin/` folder is added to your path**.


### Building and running example applications

Example applications are located in `source/examples/`. To build and run an example application, change directory to the example program you'd like to build. For example, to build and run `source/examples/video_example/video_example.cpp`, first navigate to that directory:
```
cd source/examples/video_example
```

Once you're in the correct directory, run the following commands:

```
mkdir build
cd build
cmake ..
cmake --build . --config "Release"
```

If this works, the .exe is now built and is in `/source/examples/video_example/build/Release/`. There will be additional necessary files copied to that directory as well. 

If it did not work, it likely failed to find or use your OpenCV installation while running `cmake ..`. You will need to use CMake GUI to open the example project and specify the `OpenCV_DIR` path explicitly, and then run `cmake ..` again.

For each of the example applications, see the `README.md` files in their respective directories for more information on usage.

------------------------

Visit the CU Denver [Computer Graphics and VR Lab](http://graphics.ucdenver.edu/) website for more information on this project, and other AR/VR projects from our lab.

<p align="center">
<img src="http://graphics.ucdenver.edu/img/cgvr_ucdenver.png" alt="image_example" width="500" />
</p>


