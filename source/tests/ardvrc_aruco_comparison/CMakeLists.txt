cmake_minimum_required(VERSION 3.18)

project( ardvrc_aruco_comparison )

set (CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:/Program Files (x86)/ArDVRC/include/")
include_directories("C:/Program Files (x86)/zlib/include/")
include_directories("C:/Program Files (x86)/CNPY/include/")
link_directories("C:/Program Files (x86)/ArDVRC/lib/")
link_directories("C:/Program Files (x86)/zlib/lib/")
link_directories("C:/Program Files (x86)/CNPY/lib/")

add_executable( ardvrc_aruco_comparison ardvrc_aruco_comparison.cpp )

target_link_libraries(ardvrc_aruco_comparison PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(ardvrc_aruco_comparison PRIVATE "ArDVRC")
target_link_libraries(ardvrc_aruco_comparison PRIVATE "cnpy")
target_link_libraries(ardvrc_aruco_comparison PRIVATE "zlib")

configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
