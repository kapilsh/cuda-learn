cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_learn_binary)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.h")

# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/dev/git/cuda-learn/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/local/cuda)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(saxpy-main ${CMAKE_SOURCE_DIR}/src/saxpy_main.cc ${gpu_source_files})
target_link_libraries(saxpy-main "${TORCH_LIBRARIES}")
# NOTES
# needed to pass -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-std=c++17 -DCMAKE_CUDA_COMPILER=/home/ksharma/local/cuda/bin/nvcc
