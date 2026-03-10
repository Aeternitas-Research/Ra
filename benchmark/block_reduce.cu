#include "ra/benchmark.cuh"
#include "ra/test.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thrust/device_vector.h>

RA_TEST_MAIN(argc, argv);

template <typename T, int block_size_x>
void
benchmark_block_reduce(const size_t n) {
  const int n_block = cuda::ceil_div(n, block_size_x);

  using vector = thrust::device_vector<T>;
  vector data(n, 1);
  vector result(1);

  using thrust::raw_pointer_cast;
  ra::benchmark::block_reduce<T, block_size_x><<<n_block, block_size_x>>>(
    raw_pointer_cast(data.data()), raw_pointer_cast(result.data()), n);

  REQUIRE(result[0] == static_cast<T>(n));
}

TEST_CASE("block_reduce", "[!benchmark]") {
  BENCHMARK("int, 128, 1<<20") { benchmark_block_reduce<int, 128>(1 << 20); };
  BENCHMARK("int, 128, 1<<30") { benchmark_block_reduce<int, 128>(1 << 30); };
  BENCHMARK("int, 256, 1<<20") { benchmark_block_reduce<int, 256>(1 << 20); };
  BENCHMARK("int, 256, 1<<30") { benchmark_block_reduce<int, 256>(1 << 30); };
  BENCHMARK("int, 512, 1<<20") { benchmark_block_reduce<int, 512>(1 << 20); };
  BENCHMARK("int, 512, 1<<30") { benchmark_block_reduce<int, 512>(1 << 30); };
  BENCHMARK("double, 128, 1<<20") {
    benchmark_block_reduce<double, 128>(1 << 20);
  };
  BENCHMARK("double, 128, 1<<30") {
    benchmark_block_reduce<double, 128>(1 << 30);
  };
  BENCHMARK("double, 256, 1<<20") {
    benchmark_block_reduce<double, 256>(1 << 20);
  };
  BENCHMARK("double, 256, 1<<30") {
    benchmark_block_reduce<double, 256>(1 << 30);
  };
  BENCHMARK("double, 512, 1<<20") {
    benchmark_block_reduce<double, 512>(1 << 20);
  };
  BENCHMARK("double, 512, 1<<30") {
    benchmark_block_reduce<double, 512>(1 << 30);
  };
}
