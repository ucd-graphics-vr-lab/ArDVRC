cmake_minimum_required(VERSION 3.18)

project( video_example )

#######################################################################################
# Find OpenCV (make sure it is installed on your system and built with contrib modules)
#######################################################################################
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


#######################################################################################
# Set the include and link directories for ArDVRC
#######################################################################################
include_directories("../../../include/")
link_directories("../../../lib/")


#######################################################################################
# Executable is video_example.cpp
#######################################################################################
add_executable( video_example video_example.cpp )


#######################################################################################
# Required link libraries for this application
#######################################################################################
target_link_libraries(video_example PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(video_example PRIVATE "ArDVRC")


#######################################################################################
# Resources
#######################################################################################

# TODO: Instead of specifying relative paths to resources in the .cpp, this is
# currently set up to copy the necessary files into the same location as the .exe, such
# as the custom dictionary and images. Since this just copies stuff to the build
# directory, these copies get deleted when you remove the build directory. However, 
# this is probably not a good idea for projects that require a lot of resources. Will 
# research the best way to accomplish what I'm trying to do and modify this. For now, 
# if you plan to use a lot of images or large videos, please us correct paths in the 
# .cpp file and avoid using the below copying method.

# Copies the dictionary into the same directory as the executable
# Copies example images used in image_example.cpp into the executable directory

if (WIN32)
    configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
    configure_file("../../../data/sample_data/example_video.mp4" "./Release/example_video.mp4" COPYONLY)
endif (WIN32)

if (UNIX)
    configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "${CMAKE_BINARY_DIR}/DICT_4X4_64_ENTROPY.npy" COPYONLY)
    configure_file("../../../data/sample_data/example_video.mp4" "${CMAKE_BINARY_DIR}/example_video.mp4" COPYONLY)
endif (UNIX)
