cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_learn_binary)

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
add_executable(saxpy_main ${CMAKE_SOURCE_DIR}/src/saxpy_main.cc ${gpu_source_files})
target_link_libraries(saxpy_main "${TORCH_LIBRARIES}")

# benchmarks
add_executable(benchmark_matmul_naive ${CMAKE_SOURCE_DIR}/src/benchmarks/benchmark_matmul_naive.cpp ${gpu_source_files})
target_link_libraries(benchmark_matmul_naive "${TORCH_LIBRARIES}")

# tests
add_executable(test_matmul ${CMAKE_SOURCE_DIR}/src/tests/test_matmul.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_matmul "${TORCH_LIBRARIES}")

add_executable(test_vector_add ${CMAKE_SOURCE_DIR}/src/tests/test_vector_add.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_vector_add "${TORCH_LIBRARIES}")

add_executable(test_image_blur ${CMAKE_SOURCE_DIR}/src/tests/test_image_blur.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_image_blur "${TORCH_LIBRARIES}")

add_executable(test_cuda_capabilities ${CMAKE_SOURCE_DIR}/src/tests/test_cuda_capabilities.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_cuda_capabilities "${TORCH_LIBRARIES}")

enable_testing()

add_test(NAME test_matmul COMMAND test_matmul)
add_test(NAME test_vector_add COMMAND test_vector_add)
add_test(NAME test_image_blur COMMAND test_image_blur)
add_test(NAME test_cuda_capabilities COMMAND test_cuda_capabilities)

# NOTES
# needed to pass -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-std=c++17 -DCMAKE_CUDA_COMPILER=/home/ksharma/local/cuda/bin/nvcc
