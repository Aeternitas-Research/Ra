#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

__host__ Error
PMesh1D::read() {
  ra_invoke(local.read(config.topology.rank.self));

  mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  return cudaSuccess;
}

} // namespace ra
