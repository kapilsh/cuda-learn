#include <iostream>
#include "gpu.hpp"
#include <torch/torch.h>

int main()
{
    std::cout << "Hello, world!" << std::endl;

    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();

    std::vector<int64_t> sizes = {2, 3};
    auto tensor = torch::randn(sizes);
    std::cout << "Tensor:\n" << tensor << '\n';
    return 0;
}
