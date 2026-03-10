#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

Error
PMesh1D::copy(const PMesh1D& other) {
  config = other.config;
  ra_invoke(local.copy(other.local));

  ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  return cudaSuccess;
}

} // namespace ra
