cmake_minimum_required(VERSION 3.10)
project(CatBreedDetector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/onnxruntime)

add_executable(cat_detector src/main.cpp)
target_link_libraries(cat_detector ${OpenCV_LIBS} onnxruntime)
