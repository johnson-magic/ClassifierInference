cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(ClassifierInference)
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

# 查找 OpenCV 包

find_package(OpenCV REQUIRED)

# 设置可执行文件名
set(SOURCES
    src/main.cpp
    src/classifier_inferencer.cpp
    src/utils.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES}) # main.cpp 为你的源文件名称
include_directories("./include")

# 链接 OpenCV 库和 ONNX Runtime 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} "C:/Users/24601/Desktop/fabu/software/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/lib/onnxruntime.lib" "C:/Users/24601/Desktop/fabu/software/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/lib/onnxruntime_providers_shared.lib")
# 包含OpenCV头文件和ONNX Runtime头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("C:/Users/24601/Desktop/fabu/software/onnxruntime-win-x64-1.15.1/onnxruntime-win-x64-1.15.1/include")

# 设置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
