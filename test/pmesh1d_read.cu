#include "ra/pmesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::read", "[pmesh]") {
  using ra::MeshConfig;
  using ra::PMesh1D;

  MeshConfig config = {
    .name = "test.PMesh1D",
    .file =
      {
        .handle = "read",
        .directory = "./output-test/",
      },
    .geometry =
      {
        .dof = 3,
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

  m1.local.config.info.step = 1;
  m1.local.config.info.time = 1.0;

  auto r = m1.write();
  REQUIRE(r == cudaSuccess);

  m1.local.config.info.step = 2;
  m1.local.config.info.time = 2.0;

  r = m1.read();
  REQUIRE(r == cudaSuccess);
  REQUIRE(m1.local.config.info.step == 1);

  using Catch::Matchers::WithinRel;
  REQUIRE_THAT(m1.local.config.info.time, WithinRel(1.0, 1e-14));
}
