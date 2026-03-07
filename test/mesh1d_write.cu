#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/generate.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::write", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .file =
      {
        .handle    = "write",
        .directory = "./output-test/",
      },
    .geometry =
      {
        .dof         = 2,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  m1.config.info.step = 1;
  m1.config.info.time = 1.0;

  thrust::generate(m1.host.x.begin(), m1.host.x.end(), [&] { return 1.0; });
  thrust::generate(m1.host.f.begin(), m1.host.f.end(), [&] { return 2.0; });

  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);

  const auto r = m1.write(mpi_rank);
  REQUIRE(r == cudaSuccess);
}
