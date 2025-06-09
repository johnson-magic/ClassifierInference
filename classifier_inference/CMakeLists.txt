cmake_minimum_required(VERSION 3.10)
find_package(OpenCV REQUIRED)
set(CLASSIFIER_INFERENCE_LIBRARY
    src/classifier_inferencer.cpp
)
add_library(classifier_inference ${CLASSIFIER_INFERENCE_LIBRARY})

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
include_directories("./include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIR})
#include_directories("D:/software_path/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/include")

set_target_properties(classifier_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)  # 这个是必须的，否则会报错
