cmake_minimum_required(VERSION 2.8.12)

project(validator)
add_subdirectory(./dlib dlib_build)

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      add_executable(validator web_cam_validator.cpp)
      target_link_libraries(validator dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found/")
   endif()
endif()


