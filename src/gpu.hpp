#include <torch/types.h>

void printCudaVersion();

torch::Tensor saxpy_wrapper(const torch::Tensor& x, torch::Tensor y, float a);
