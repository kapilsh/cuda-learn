#include "imports/catch_amalgamated.hpp"
#include <torch/torch.h>
#include "../vector_add.cuh"

TEST_CASE("VECTOR ADD", "[basic]") {
    torch::manual_seed(42);
    const auto x = torch::randn({100, 400}, torch::kCUDA);
    const auto y = torch::randn({100, 400}, torch::kCUDA);
    const auto z = vector_add(x, y);
    const auto z_actual = x + y;
    REQUIRE(torch::allclose(z_actual, z, .0001, .0001));
    const auto sum_value = torch::sum(z);
    REQUIRE(abs(sum_value.item<float>() + 147.38) < 0.0001);
}