cmake_minimum_required(VERSION 3.18)

project( image_example )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("C:/Program Files (x86)/ArDVRC/include/")
link_directories("C:/Program Files (x86)/ArDVRC/lib/")

add_executable( image_example image_example.cpp )

target_link_libraries(image_example PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(image_example PRIVATE "ArDVRC")

configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
configure_file("../../../data/sample_data/example_image1.jpg" "./Release/example_image1.jpg" COPYONLY)
configure_file("../../../data/sample_data/example_image2.jpg" "./Release/example_image2.jpg" COPYONLY)
configure_file("../../../data/sample_data/example_image3.jpg" "./Release/example_image3.jpg" COPYONLY)
configure_file("../../../data/sample_data/example_image4.jpg" "./Release/example_image4.jpg" COPYONLY)

