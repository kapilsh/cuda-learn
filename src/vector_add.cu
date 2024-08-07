//
// Created by ksharma on 8/7/24.
//

#include "vector_add.cuh"
#include <cuda_runtime.h>

__global__ void vector_add_kernel(int n, const float* x, const float* y, float* z) {
    if (const auto i = blockIdx.x*blockDim.x + threadIdx.x; i < n) {
        z[i] = x[i] + y[i];
    }
}


torch::Tensor vector_add(const torch::Tensor& x, const torch::Tensor& y) {
    constexpr int32_t block_size = 1024;
    auto n = static_cast<int32_t>(torch::numel(x));
    const int32_t num_blocks = (n + block_size - 1) / block_size;
    auto z = torch::empty_like(x);
    vector_add_kernel<<<num_blocks, block_size>>>(n, x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>());
    cudaDeviceSynchronize();
    return z;
}