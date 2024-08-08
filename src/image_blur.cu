//
// Created by ksharma on 8/8/24.
//

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "image_blur.cuh"


__global__ void convolution_naive_kernel(const uint8_t* input, uint8_t* output, const int32_t width,
                                         const int32_t height, const int32_t kernel_size)
{
    int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    const auto kernel_width = kernel_size / 2;

    if (row < height && column < width)
    {
        int32_t sum = 0.0f;
        int32_t pixels = 0;

        // below is the convolution kernel for a 2D image
        for (auto i = -kernel_width; i <= kernel_width; ++i)
        {
            for (auto j = -kernel_width; j <= kernel_width; ++j)
            {
                const auto current_row = row + i;
                const auto current_column = column + j;
                if (current_row >= 0 && current_row < height && current_column >= 0 && current_column < width)
                {
                    sum += input[current_row * width + current_column];
                    ++pixels;
                }
            }
        }
        output[row * width + column] = static_cast<int8_t>(sum / pixels);
    }
}

torch::Tensor image_blur(const torch::Tensor& input, const int32_t kernel_size)
{
    assert(kernel_size % 2 == 1 && "Kernel size must be odd");

    constexpr int32_t block_size = 32;

    const auto height = input.sizes()[0];
    const auto width = input.sizes()[1];

    dim3 gridDim((height + block_size - 1) / block_size, (width + block_size - 1) / block_size);
    dim3 blockDim(block_size, block_size);

    auto output = torch::empty({height, width}, input.options());

    convolution_naive_kernel<<<gridDim, blockDim>>>(input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), width,
                                                    height, kernel_size);
    return output;
}
