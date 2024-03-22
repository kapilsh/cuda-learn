cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_learn_binary)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.hpp")

find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
add_executable(cuda_learn_binary ${cpu_source_files} ${gpu_source_files})

