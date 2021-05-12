cmake_minimum_required(VERSION 3.18)

project( video_example )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:/Program Files (x86)/ArDVRC/include/")
link_directories("C:/Program Files (x86)/ArDVRC/lib/")

add_executable( video_example video_example.cpp )

target_link_libraries(video_example PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(video_example PRIVATE "ArDVRC")

configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
configure_file("../../../data/sample_data/example_video.mp4" "./Release/example_video.mp4" COPYONLY)

