# Image Example

This application processes individual images using ArDVRC and stitches them together.

<img src="http://graphics.ucdenver.edu/img/ardvrc_image_example.png" alt="image_example" width="600" />


### Building image_example.exe

Run the following commands in the terminal (this assumes OpenCV is found without problems):
```
mkdir build
cd build
cmake ..
cmake --build . --config "Release"
```

### Running image_example.exe

The .exe is now built and is in the `Release/` folder inside your current (build) directory. There are additional files copied to that directory as well, such as a dictionary NPY file and the input sample images.

Run the following commands in the terminal to run the program with the default example images provided:

```
cd Release
./image_example.exe
```

### TODO

Allow command line arguments for images to process.

------------------------

Visit http://graphics.ucdenver.edu/ for more information on this project, and other AR/VR projects from
CU Denver's Computer Graphics and VR Lab.

<p align="center">
<img src="http://graphics.ucdenver.edu/img/cgvr_ucdenver.png" alt="image_example" width="500" />
</p>


