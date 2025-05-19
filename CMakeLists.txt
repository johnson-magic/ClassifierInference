cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(classifier_demo)

set(CMAKE_BUILD_TYPE Release)
option(SPEED_TEST "SPEED TEST" OFF)
option(CONFORMANCE_TEST "conformance test" OFF)
option(ENCRYPT "encrypt" OFF)
if(SPEED_TEST)
    add_definitions(-DSPEED_TEST)
endif()
if(CONFORMANCE_TEST)
    add_definitions(-DCONFORMANCE_TEST)
endif()
if(ENCRYPT)
    add_definitions(-DENCRYPT)
endif()

# 设置可执行文件名
set(CLASSIFIER_DEMO_SOURCES
    demo/main.cpp
    demo/utils/src/utils.cpp
)

add_subdirectory("./classifier_inference")
add_executable(${PROJECT_NAME} ${CLASSIFIER_DEMO_SOURCES})
include_directories("./demo/utils/include")
include_directories("./classifier_inference/include")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("D:/software_path/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/include")

target_link_libraries(${PROJECT_NAME}
                      classifier_inference
                      ${OpenCV_LIBS}
                      "D:/software_path/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/lib/onnxruntime.lib"
                      "D:/software_path/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/lib/onnxruntime_providers_shared.lib")

# 设置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)  # 这个是必须的，否则会报错
