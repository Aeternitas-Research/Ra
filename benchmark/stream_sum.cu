#include "ra/benchmark.cuh"
#include "ra/test.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

RA_TEST_MAIN(argc, argv);

template <typename T>
void
benchmark_stream_sum(const std::size_t n) {
  using vector = thrust::device_vector<T>;
  vector r(n);
  vector x1(n, static_cast<T>(1));
  vector x2(n, static_cast<T>(2));
  vector x3(n, static_cast<T>(3));
  vector x4(n, static_cast<T>(4));

  ra::benchmark::stream_sum(r, x1, x2, x3, x4);
  thrust::host_vector<T> result = r;

  using Catch::Matchers::WithinRel;
  thrust::for_each(result.begin(), result.end(), [&](double value) {
    REQUIRE_THAT(value, WithinRel(10.0, 1e-14));
  });
}

TEST_CASE("stream_sum", "[!benchmark]") {
  BENCHMARK("int, 1<<20") { benchmark_stream_sum<int>(1 << 20); };
  BENCHMARK("int, 1<<25") { benchmark_stream_sum<int>(1 << 25); };
  BENCHMARK("double, 1<<20") { benchmark_stream_sum<double>(1 << 20); };
  BENCHMARK("double, 1<<25") { benchmark_stream_sum<double>(1 << 25); };
}
