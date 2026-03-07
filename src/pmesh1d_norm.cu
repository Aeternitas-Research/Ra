#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <cuda/std/cmath>
#include <mpi.h>

namespace ra {

__host__ Error
PMesh1D::norm(OperationSpace space, double& r, const std::string type) {
  ra_invoke(local.norm(space, r, type));

  mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  if ((type == "1") || (type == "l1") || (type == "l^1")) {
    mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  } else if ((type == "2") || (type == "l2") || (type == "l^2")) {
    r *= r;

    mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

    r = cuda::std::sqrt(r);
  } else if ((type == "infinity") || (type == "inf") || (type == "max")) {
    mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
