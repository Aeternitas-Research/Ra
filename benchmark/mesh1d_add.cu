#include "ra/benchmark.cuh"
#include "ra/test.cuh"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thrust/for_each.h>

RA_TEST_MAIN(argc, argv);

__host__ void
benchmark_mesh1d_add(ra::OperationSpace space, const size_t n) {
  using Config  = ra::MeshConfig;
  Config config = {
    .name = "benchmark.Mesh1D",
    .geometry =
      {
        .dof         = 2,
        .extent      = {n, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };

  using Mesh = ra::Mesh1D;
  Mesh y(config);
  Mesh c(config);
  Mesh x(config);

  y.assign(space, 1.0);
  c.assign(space, 2.0);
  x.assign(space, 3.0);
  ra::benchmark::mesh1d_add(space, y, c, x);

  if (space == ra::OperationSpace::Device) {
    ra_invoke(y.transfer(cudaMemcpyDeviceToHost, false, true));
    ra_invoke(c.transfer(cudaMemcpyDeviceToHost, false, true));
    ra_invoke(x.transfer(cudaMemcpyDeviceToHost, false, true));
  }

  using Catch::Matchers::WithinRel;
  thrust::for_each(y.host.f.begin(), y.host.f.end(), [&](double value) {
    REQUIRE_THAT(value, WithinRel(7.0, 1e-14));
  });
  thrust::for_each(c.host.f.begin(), c.host.f.end(), [&](double value) {
    REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
  });
  thrust::for_each(x.host.f.begin(), x.host.f.end(), [&](double value) {
    REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
  });
}

TEST_CASE("Mesh1D::add", "[!benchmark]") {
  const auto host   = ra::OperationSpace::Host;
  const auto device = ra::OperationSpace::Device;

  BENCHMARK("Host, 1<<10") { benchmark_mesh1d_add(host, 1 << 10); };
  BENCHMARK("Device, 1<<10") { benchmark_mesh1d_add(device, 1 << 10); };
  BENCHMARK("Host, 1<<20") { benchmark_mesh1d_add(host, 1 << 20); };
  BENCHMARK("Device, 1<<20") { benchmark_mesh1d_add(device, 1 << 20); };
}
