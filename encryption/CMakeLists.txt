cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(timelimit)
set(CMAKE_BUILD_TYPE Release)

set(SOURCES
    time_limit.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES}) # main.cpp 为你的源文件名称
include_directories("C:/Users/24601/Desktop/fabu/code/PrecisionAngleDetectionInference/cpp/tools")


# 设置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
