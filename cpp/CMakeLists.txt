cmake_minimum_required(VERSION 3.16)
project(rdlnet_tflite_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找 OpenCV 库
# REQUIRED 表示如果找不到 OpenCV，CMake 配置将报错并停止
find_package(OpenCV REQUIRED)

# Explicit overrides (highest priority)
set(TFLITE_INCLUDE_DIR "/Users/chenlong/workspace/tensorflow")
set(TFLITE_LIBRARY "${TFLITE_INCLUDE_DIR}/bazel-bin/tensorflow/lite/libtensorflowlite.dylib")

add_executable(rdlnet_tflite_infer rdlnet_tflite_infer.cc)
target_include_directories(rdlnet_tflite_infer PRIVATE 
  ${TFLITE_INCLUDE_DIR}
  ${TFLITE_INCLUDE_DIR}/bazel-tensorflow/external/flatbuffers/include
  ${TFLITE_INCLUDE_DIR}/bazel-tensorflow/external/com_google_absl
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rdlnet_tflite_infer PRIVATE ${TFLITE_LIBRARY} ${OpenCV_LIBS})

