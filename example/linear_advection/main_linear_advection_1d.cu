#include "ra/mpi.cuh"
#include <mpi.h>

int
main(int argc, char* argv[]) {
  ra_mpi_invoke(MPI_Init(&argc, &argv));

  // assign device
  int mpi_rank_local{};
  {
    int mpi_rank{};
    ra_mpi_invoke(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    MPI_Comm communicator{};
    ra_mpi_invoke(MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL,
      &communicator));
  }

  return 0;
}
