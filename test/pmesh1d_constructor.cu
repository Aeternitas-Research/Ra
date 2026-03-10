#include "ra/pmesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::PMesh1D", "[pmesh]") {
  using ra::MeshConfig;
  using ra::PMesh1D;

  PMesh1D m1{};

  MeshConfig config = {
    .name = "test.PMesh1D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
  };
  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
  int mpi_extent[6];
  REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, mpi_extent) == MPI_SUCCESS);
  PMesh1D m2(mpi_rank, mpi_extent, config);

  REQUIRE(m2.config.global.name == config.name);
  REQUIRE(m2.config.global.file.handle == config.file.handle);
  REQUIRE(m2.config.global.file.directory == config.file.directory);
  REQUIRE(m2.config.topology.rank.self == mpi_rank);
  for (int d = 0; d < 1; ++d) {
    REQUIRE(m2.config.topology.extent[d] == mpi_extent[d]);
  }
}
