#include "ra/pmesh.cuh"
#include "ra/test.cuh"
#include <thrust/generate.h>
#include <thrust/random.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::transfer", "[pmesh]") {
  using ra::MeshConfig;
  using ra::PMesh1D;

  MeshConfig config = {
    .name = "test.PMesh1D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
    .geometry =
      {
        .element =
          {
            .dof = 3,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{5, 7}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
  int mpi_extent[6];
  REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, mpi_extent) == MPI_SUCCESS);
  PMesh1D m1(mpi_rank, mpi_extent, config);
  m1.calibrate();

  thrust::default_random_engine rng(7);
  thrust::uniform_real_distribution<double> dist{};
  thrust::generate(
    m1.local.host.x.begin(), m1.local.host.x.end(), [&] { return dist(rng); });
  thrust::generate(
    m1.local.host.f.begin(), m1.local.host.f.end(), [&] { return dist(rng); });

  auto r = m1.transfer(cudaMemcpyHostToDevice, false, true);
  REQUIRE(r == RA_SUCCESS);

  r = m1.transfer(cudaMemcpyDeviceToHost, false, true);
  REQUIRE(r == RA_SUCCESS);
}
