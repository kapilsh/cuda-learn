#pragma once

#include <torch/torch.h>

torch::Tensor image_blur(const torch::Tensor& input, int32_t kernel_size);