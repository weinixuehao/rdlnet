cmake_minimum_required(VERSION 3.16)
project(rdlnet_tflite_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找 OpenCV 库
# REQUIRED 表示如果找不到 OpenCV，CMake 配置将报错并停止
# find_package(OpenCV REQUIRED)

# TensorFlow Lite headers + library
# - macOS: provide local defaults
# - Linux: pass via cmake -DTFLITE_INCLUDE_DIR=... -DTFLITE_LIBRARY=...
set(TFLITE_INCLUDE_DIR "" CACHE PATH "Path to TensorFlow repo (for TFLite headers)")
set(TFLITE_LIBRARY "" CACHE FILEPATH "Path to TensorFlow Lite library (libtensorflowlite.*)")

if(APPLE)
  if(NOT TFLITE_INCLUDE_DIR)
    set(TFLITE_INCLUDE_DIR "/Users/chenlong/workspace/tensorflow" CACHE PATH "" FORCE)
  endif()
  if(NOT TFLITE_LIBRARY AND TFLITE_INCLUDE_DIR)
    set(TFLITE_LIBRARY "${TFLITE_INCLUDE_DIR}/bazel-bin/tensorflow/lite/libtensorflowlite.dylib" CACHE FILEPATH "" FORCE)
  endif()
endif()

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY)
  message(FATAL_ERROR "Set -DTFLITE_INCLUDE_DIR=/path/to/tensorflow -DTFLITE_LIBRARY=/path/to/libtensorflowlite.(a|so|dylib)")
endif()

add_executable(rdlnet_tflite_infer rdlnet_tflite_infer.cc)
target_include_directories(rdlnet_tflite_infer PRIVATE 
  ${TFLITE_INCLUDE_DIR}
  ${TFLITE_INCLUDE_DIR}/bazel-tensorflow-2.21.0/external/flatbuffers/include
  ${TFLITE_INCLUDE_DIR}/bazel-tensorflow-2.21.0/external/com_google_absl
  ${OpenCV_INCLUDE_DIRS}
)
set(opencv_core ${OpenCV_LIBS}/libopencv_core.so.409)
set(opencv_imgcodecs ${OpenCV_LIBS}/libopencv_imgcodecs.so.409)
set(opencv_imgproc ${OpenCV_LIBS}/libopencv_imgproc.so.409)
target_link_libraries(rdlnet_tflite_infer PRIVATE ${opencv_core} ${opencv_imgcodecs} ${opencv_imgproc} ${TFLITE_LIBRARY})

