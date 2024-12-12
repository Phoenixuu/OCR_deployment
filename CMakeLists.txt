cmake_minimum_required(VERSION 3.14)
project(OCRProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Add ONNX Runtime
find_package(onnxruntime REQUIRED)

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")

# Build target
add_executable(OCRApp ${SOURCES})
target_link_libraries(OCRApp PRIVATE onnxruntime)