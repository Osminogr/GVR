cmake_minimum_required(VERSION 2.8.12)

project(validator)
add_subdirectory(./dlib dlib_build)

add_executable(validator validator.cpp)
target_link_libraries(validator dlib::dlib)
