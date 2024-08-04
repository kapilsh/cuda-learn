#include <iostream>
#include "saxpy.cuh"
#include "utils.cuh"
#include <torch/torch.h>

int main()
{
    std::cout << "Hello, world!" << std::endl;
    printCudaVersion();
    std::vector<int64_t> sizes = {2, 3};
    const auto x = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor x:\n" << x << '\n';
    const auto y = torch::randn(sizes, torch::kCUDA);
    std::cout << "Tensor y:\n" << y << '\n';
    saxpy_wrapper(x, y, 2.0);
    std::cout << "Out:\n" << y << '\n';
    return 0;
}
