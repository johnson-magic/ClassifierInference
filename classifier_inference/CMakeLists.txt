cmake_minimum_required(VERSION 3.10)
# 设置项目名称
project(classifier_inference_library)

set(CLASSIFIER_INFERENCE_LIBRARY
    src/classifier_inference.cpp
)
add_library(classifier_inference ${CLASSIFIER_INFERENCE_LIBRARY})

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
include_directories("./include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIR})

set_target_properties(classifier_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)  # 这个是必须的，否则会报错

set_target_properties(classifier_inference PROPERTIES PUBLIC_HEADER "include/classifier_inference.h")
install(TARGETS classifier_inference)
