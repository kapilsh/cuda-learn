#include <iostream>
#include "gpu.hpp"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main()
{
    std::cout << "Hello, world!" << std::endl;

    printCudaVersion();

    std::vector<int64_t> sizes = {2, 3};
    auto x = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor x:\n" << x << '\n';    auto y = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor y:\n" << y << '\n';

    saxpy_wrapper(6, x.data_ptr<float>(), y.data_ptr<float>(), 2.0);

    std::cout << "Out:\n" << y << '\n';

    return 0;
}
