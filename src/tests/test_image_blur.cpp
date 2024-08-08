#include "imports/catch_amalgamated.hpp"
#include <torch/torch.h>
#include "../image_blur.cuh"

TEST_CASE("CONVOLUTION", "[basic]")
{
    torch::manual_seed(42);
    const auto x = torch::randint(0, 256, {3, 3}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    constexpr int32_t kernel_size = 3;
    const auto output = image_blur(x, kernel_size);
    const auto expected_output = torch::tensor({{95, 128, 127}, {81, 98, 94}, {75, 89, 74}},
                                               torch::dtype(torch::kUInt8).device(torch::kCUDA));
    REQUIRE(torch::equal(output, expected_output));
}
