#include <torch/torch.h>
#include "../matmul.cuh"
#include <cuda_runtime.h>

int main() {
    at::globalContext().setFloat32MatmulPrecision("high");
    for (int i = 0; i < 16; i++) {
        const auto size = static_cast<int32_t>(std::pow(2, i));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        std::vector<int64_t> sizes = {size, size};
        const auto x = torch::randn(sizes, torch::kCUDA);
        const auto y = torch::randn(sizes, torch::kCUDA);
        // std::cout << "dtype" << x.dtype() << "\n";
        
        cudaEventRecord(start);
        auto output = matmul_naive(x, y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        const auto size_casted = static_cast<double>(size);
        const auto tflops = 2 * size_casted* size_casted * size_casted * 10e-12 / (milliseconds * .0001);
        std::cout << "[" << size << "x" << size << "] Time taken: "  << milliseconds << "ms | " "TFLOPS: " << tflops  << "\n";
    }
    return 0;
}