cmake_minimum_required(VERSION 3.18)

project( image_example )

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
# Executable is image_example.cpp
#######################################################################################
add_executable( image_example image_example.cpp )


#######################################################################################
# Required link libraries for this application
#######################################################################################
target_link_libraries(image_example PRIVATE "${OpenCV_LIBS}" )
target_link_libraries(image_example PRIVATE "ArDVRC")


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


if (WIN32)

    # Copies dictionary to executable directory
    configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "./Release/DICT_4X4_64_ENTROPY.npy" COPYONLY)
    
    # Copies example images used in image_example.cpp into the executable directory
    configure_file("../../../data/sample_data/example_image1.jpg" "./Release/example_image1.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image2.jpg" "./Release/example_image2.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image3.jpg" "./Release/example_image3.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image4.jpg" "./Release/example_image4.jpg" COPYONLY)

endif (WIN32)

if (UNIX)
    configure_file("../../../data/entropy_dictionary/DICT_4X4_64_ENTROPY.npy" "${CMAKE_BINARY_DIR}/DICT_4X4_64_ENTROPY.npy" COPYONLY)
    
    # Copies example images used in image_example.cpp into the executable directory
    configure_file("../../../data/sample_data/example_image1.jpg" "${CMAKE_BINARY_DIR}/example_image1.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image2.jpg" "${CMAKE_BINARY_DIR}/example_image2.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image3.jpg" "${CMAKE_BINARY_DIR}/example_image3.jpg" COPYONLY)
    configure_file("../../../data/sample_data/example_image4.jpg" "${CMAKE_BINARY_DIR}/example_image4.jpg" COPYONLY)
endif (UNIX)