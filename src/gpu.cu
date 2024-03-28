#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu.hpp"

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
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

torch::Tensor saxpy_wrapper(int n, const torch::Tensor& x, torch::Tensor y, float a) {
    saxpy<<<n, 1>>>(n, a, x.data_ptr<float>(), y.data_ptr<float>());
    std::cout <<  "Calculated saxpy\n";
    cudaDeviceSynchronize();
    return y;
}