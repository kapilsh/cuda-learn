cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_learn_binary)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.hpp")

list(APPEND CMAKE_PREFIX_PATH /home/ksharma/dev/git/cuda-learn/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/anaconda3/envs/cuda-learn)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(cuda_learn_binary ${cpu_source_files} ${gpu_source_files})
target_link_libraries(cuda_learn_binary "${TORCH_LIBRARIES}")
