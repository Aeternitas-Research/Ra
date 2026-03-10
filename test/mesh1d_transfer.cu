#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/generate.h>
#include <thrust/random.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::transfer", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
    .geometry =
      {
        .element =
          {
            .dof = 2,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  thrust::default_random_engine rng(7);
  thrust::uniform_real_distribution<double> dist{};
  thrust::generate(
    m1.host.x.begin(), m1.host.x.end(), [&] { return dist(rng); });
  thrust::generate(
    m1.host.f.begin(), m1.host.f.end(), [&] { return dist(rng); });

  auto r = m1.transfer(cudaMemcpyHostToDevice, false, true);
  REQUIRE(r == cudaSuccess);

  r = m1.transfer(cudaMemcpyDeviceToHost, false, true);
  REQUIRE(r == cudaSuccess);
}
