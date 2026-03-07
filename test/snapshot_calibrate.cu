#include "ra/snapshot.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Snapshot::calibrate", "[snapshot]") {
  using ra::Snapshot;
  using ra::SnapshotConfig;

  SnapshotConfig config = {
    .name = "test.Snapshot",
    .time =
      {
        .start = 0.0,
        .stop  = 1.0,
        .now   = 0.5,
        .delta = 0.1,
      },
  };
  Snapshot s(config);
  REQUIRE(s.calibrate() == cudaSuccess);

  auto s_config = s.config.get();

  REQUIRE(s_config->mpi.initialized == true);

  int mpi_size{};
  REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) == MPI_SUCCESS);
  REQUIRE(mpi_size == s_config->mpi.size);

  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
  REQUIRE(mpi_rank == s_config->mpi.rank);
}
