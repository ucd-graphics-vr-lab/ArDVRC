cmake_minimum_required(VERSION 3.18)

project( ardvrc_aruco_comparison )

set (CMAKE_CXX_STANDARD 14)

#######################################################################################
# Find OpenCV (make sure it is installed on your system and built with contrib modules)
#######################################################################################
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#######################################################################################
# Set the include and link directories
#######################################################################################

# ArDVRC, zlib and CNPY are required
include_directories("../../../ArDVRC/include/")
include_directories("C:/Program Files (x86)/zlib/include/")
include_directories("C:/Program Files (x86)/CNPY/include/")
link_directories("../../../ArDVRC/lib/")
link_directories("C:/Program Files (x86)/zlib/lib/")
link_directories("C:/Program Files (x86)/CNPY/lib/")

#######################################################################################
# Executable is ardvrc_aruco_comparison.cpp
#######################################################################################
add_executable( ardvrc_aruco_comparison ardvrc_aruco_comparison.cpp )

#######################################################################################
# Required link libraries for this application
#######################################################################################
target_link_libraries(ardvrc_aruco_comparison PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(ardvrc_aruco_comparison PRIVATE "ArDVRC")
target_link_libraries(ardvrc_aruco_comparison PRIVATE "cnpy")
target_link_libraries(ardvrc_aruco_comparison PRIVATE "zlib")

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
configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
