#pragma once
#include <torch/torch.h>

torch::Tensor matmul_naive(const torch::Tensor& matrix1, const torch::Tensor& matrix2);