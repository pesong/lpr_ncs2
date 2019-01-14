cmake_minimum_required(VERSION 3.12)
project(License_Plate_Recognition_ros_ncsv2)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/perception_cv_ncs/include)

add_executable(License_Plate_Recognition_ros_ncsv2
        src/perception_cv_ncs/include/ncs_util.h
        src/perception_cv_ncs/include/lpr_ncs2.hpp
        src/perception_cv_ncs/include/stb_image.h
        src/perception_cv_ncs/include/stb_image_resize.h
        src/perception_cv_ncs/include/stb_image_write.h
        src/perception_cv_ncs/src/ncs_util.cpp
        src/perception_cv_ncs/src/lpr_ncs2.cpp
        src/perception_cv_ncs/main.cc)
