#include "ra/pmesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::write", "[pmesh]") {
  using ra::MeshConfig;
  using ra::PMesh1D;

  MeshConfig config = {
    .name = "test.PMesh1D",
    .file =
      {
        .handle = "write",
        .directory = "./output-test/",
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

  const auto r = m1.write();
  REQUIRE(r == RA_SUCCESS);
}
