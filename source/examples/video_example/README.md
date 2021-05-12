# Video Example

<img src="http://graphics.ucdenver.edu/img/ardvrc_video_example.gif" alt="video_example" width="600" />

### Building video_example.exe

Run the following commands in the terminal (this assumes OpenCV is found without problems):
```
mkdir build
cd build
cmake ..
cmake --build . --config "Release"
```

### Running video_example.exe

The .exe is now built and is in the `Release/` folder inside your current (build) directory. There are additional files copied to that directory as well, such as a dictionary NPY file and a sample video.

Run the following commands in the terminal to run the program with the default example video provided:

```
cd Release
./video_example.exe
```

### Command line arguments

##### Use live video
```
./video_example.exe --live
```

##### Use a specific video file
```
./video_example.exe --file <filename>
```

##### Show direction vectors
```
./video_example.exe --vectors
```

##### Compare side-by-side with ArUco
```
./video_example.exe --compare
```

------------------------

Visit the CU Denver [Computer Graphics and VR Lab](http://graphics.ucdenver.edu/) website for more information on this project, and other AR/VR projects from our lab.

<p align="center">
<img src="http://graphics.ucdenver.edu/img/cgvr_ucdenver.png" alt="image_example" width="500" />
</p>


