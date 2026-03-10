#include "ra/mpi.cuh"
#include "ra/snapshot.cuh"
#include <mpi.h>

namespace ra {

Error
Snapshot::calibrate() {
  auto ptr_config = config.get();

  // MPI
  int r_mpi_initialized = 0;
  ra_mpi_invoke(MPI_Initialized(&r_mpi_initialized));
  ptr_config->mpi.initialized = (r_mpi_initialized == 1);
  ra_mpi_invoke(MPI_Comm_size(MPI_COMM_WORLD, &(ptr_config->mpi.size)));
  ra_mpi_invoke(MPI_Comm_rank(MPI_COMM_WORLD, &(ptr_config->mpi.rank)));

  return cudaSuccess;
}

} // namespace ra
