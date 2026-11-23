cmake_minimum_required(VERSION 3.18)
project(yolo_video LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV: Linux — apt/сборка из исходников; macOS — brew install opencv;
# Windows — распакованный релиз с opencv.org: -DOpenCV_DIR=C:/opencv/build
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn)
message(STATUS "OpenCV ${OpenCV_VERSION}")

add_executable(yolo_video main.cpp)
target_link_libraries(yolo_video PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(yolo_video PRIVATE /utf-8)  # русские строки в исходнике — UTF-8
endif()

# ONNX Runtime (необязательно): распакованный релиз с github.com/microsoft/onnxruntime/releases
#   cmake -DWITH_ORT=ON -DONNXRUNTIME_DIR=/path/to/onnxruntime-<os>-<arch>-<ver> ..
option(WITH_ORT "Собрать бэкенд ONNX Runtime" OFF)
set(ONNXRUNTIME_DIR "" CACHE PATH "Папка ONNX Runtime с подпапками include/ и lib/")
if(WITH_ORT)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_DIR}/include/onnxruntime
                  ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session REQUIRED)
  find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib REQUIRED)
  message(STATUS "ONNX Runtime: ${ORT_LIB}")
  target_compile_definitions(yolo_video PRIVATE WITH_ORT)
  target_include_directories(yolo_video PRIVATE ${ORT_INCLUDE_DIR})
  target_link_libraries(yolo_video PRIVATE ${ORT_LIB})
  set_target_properties(yolo_video PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
  if(WIN32)  # положить onnxruntime*.dll рядом с exe
    add_custom_command(TARGET yolo_video POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ONNXRUNTIME_DIR}/lib $<TARGET_FILE_DIR:yolo_video>)
  endif()
endif()
