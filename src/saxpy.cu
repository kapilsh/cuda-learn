#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "saxpy.h"



__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

torch::Tensor saxpy_wrapper(const torch::Tensor& x, torch::Tensor y, float a) {
    auto n = static_cast<int32_t>(torch::numel(x));
    saxpy<<<n, 1>>>(n, a, x.data_ptr<float>(), y.data_ptr<float>());
    std::cout <<  "Calculated saxpy\n";
    cudaDeviceSynchronize();
    return y;
}