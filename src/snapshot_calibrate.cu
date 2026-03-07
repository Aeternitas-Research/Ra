#include "ra/mpi.cuh"
#include "ra/snapshot.cuh"
#include <mpi.h>

namespace ra {

__host__ Error
Snapshot::calibrate() {
  auto ptr_config = config.get();

  // MPI
  int r_mpi_initialized = 0;
  mpi_invoke(MPI_Initialized(&r_mpi_initialized));
  ptr_config->mpi.initialized = (r_mpi_initialized == 1);
  mpi_invoke(MPI_Comm_size(MPI_COMM_WORLD, &(ptr_config->mpi.size)));
  mpi_invoke(MPI_Comm_rank(MPI_COMM_WORLD, &(ptr_config->mpi.rank)));

  return cudaSuccess;
}

} // namespace ra
