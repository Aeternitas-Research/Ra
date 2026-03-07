#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

__host__ Error
PMesh1D::transfer(const cudaMemcpyKind kind, const bool x, const bool f) {
  ra_invoke(local.transfer(kind, x, f));

  mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  return cudaSuccess;
}

} // namespace ra
