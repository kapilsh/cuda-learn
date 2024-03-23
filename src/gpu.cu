#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

void printCudaVersion()
{
    std::cout << "CUDA Compiled version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

void saxpy_wrapper(int n, float * x, float * y, float a)
{
    cudaPointerAttributes attributes_x{};
    auto error_x = cudaPointerGetAttributes (&attributes_x,x);
    if (error_x != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(error_x), __FILE__, __LINE__);

        exit(error_x);
    }
    cudaPointerAttributes attributes_y{};
    auto error_y = cudaPointerGetAttributes (&attributes_y, y);
    if (error_y != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(error_y), __FILE__, __LINE__);

        exit(error_y);
    }

    saxpy<<<n, 1>>>(n, a, x, y);
    std::cout <<  "Calculated saxpy\n";
    cudaDeviceSynchronize();
}