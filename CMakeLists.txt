cmake_minimum_required(VERSION 2.8.12)

SET(_NAME main)
SET(CPP ${_NAME}.cpp)

project(${_NAME})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${_NAME} ${CPP})
target_link_libraries(${_NAME} ${OpenCV_LIBS})
