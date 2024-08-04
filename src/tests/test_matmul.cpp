#include "imports/catch_amalgamated.hpp"
#include <torch/torch.h>
#include "../matmul.cuh"

TEST_CASE("MATMUL", "[basic]") {
    torch::manual_seed(42);
    const auto x = torch::randn({100, 400}, torch::kCUDA);
    const auto y = torch::randn({400, 500}, torch::kCUDA);
    const auto z = matmul_naive(x, y);
    const auto z_actual = torch::mm(x, y);
    REQUIRE(torch::allclose(z_actual, z, .0001, .0001));

    const auto sum_value = torch::sum(z);
    REQUIRE(abs(sum_value.item<float>() + 3041.39) < 0.0001);
}