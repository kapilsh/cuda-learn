cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_learn_binary)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB test_import_files "${CMAKE_SOURCE_DIR}/src/tests/imports/*")


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

add_executable(test_matmul ${CMAKE_SOURCE_DIR}/src/tests/test_matmul.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_matmul "${TORCH_LIBRARIES}")

enable_testing()

# NOTES
# needed to pass -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-std=c++17 -DCMAKE_CUDA_COMPILER=/home/ksharma/local/cuda/bin/nvcc
