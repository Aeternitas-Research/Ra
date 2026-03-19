#include "ra/pmesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::copy", "[pmesh]") {
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
            .type = ra::MeshElementType::Line,
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

  PMesh1D m2{};
  const auto r = m2.copy(m1);
  REQUIRE(r == RA_SUCCESS);
  REQUIRE(m2.config.global.name == m1.config.global.name);
  REQUIRE(m2.config.global.file.handle == m1.config.global.file.handle);
  REQUIRE(m2.config.global.file.directory == m1.config.global.file.directory);
  REQUIRE(m2.local.config.name == m1.local.config.name);
  REQUIRE(m2.local.config.file.handle == m1.local.config.file.handle);
  REQUIRE(m2.local.config.file.directory == m1.local.config.file.directory);
}
