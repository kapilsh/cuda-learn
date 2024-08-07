#include <torch/torch.h>
#include <cuda_runtime.h>
#include "matmul.cuh"

__global__ void matmul_naive_kernel(const float* input_matrix_left /*M X N*/, const float* input_matrix_right /*N X K*/,
                         float* output_matrix, int32_t M, int32_t N, int32_t K)
{
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto column = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && column < K)
    {
        auto sum = 0.0f;
        for (auto i = 0; i < N; ++i)
        {
            sum += input_matrix_left[row * N + i] * input_matrix_right[i * K + column];
        }
        output_matrix[row * K + column] = sum;
    }
}

torch::Tensor matmul_naive(const torch::Tensor& matrix1, const torch::Tensor& matrix2)
{
    constexpr int32_t block_size = 32;

    const auto M = matrix1.sizes()[0];
    const auto N = matrix1.sizes()[1];
    const auto K = matrix2.sizes()[1];

    dim3 gridDim((M + block_size - 1) / block_size, (K + block_size - 1) / block_size);
    dim3 blockDim(block_size, block_size);

    auto output_matrix = torch::empty({M, K}, matrix1.options());

    matmul_naive_kernel<<<gridDim, blockDim>>>(matrix1.data_ptr<float>(), matrix2.data_ptr<float>(),
                                               output_matrix.data_ptr<float>(), M, N, K);
    return output_matrix;
}
