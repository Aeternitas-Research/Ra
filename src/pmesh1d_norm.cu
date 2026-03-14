#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <cuda/std/cmath>
#include <mpi.h>

namespace ra {

Error
PMesh1D::norm(const OperationSpace space, double& r, const std::string type) {
  ra_invoke(local.norm(space, r, type));

  ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  if ((type == "1") || (type == "l1") || (type == "l^1")) {
    ra_mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  } else if ((type == "2") || (type == "l2") || (type == "l^2")) {
    r *= r;

    ra_mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

    r = cuda::std::sqrt(r);
  } else if ((type == "infinity") || (type == "inf") || (type == "max")) {
    ra_mpi_invoke(
      MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  } else {
    return RA_ERROR(ErrorValue::InvalidParameter);
  }

  return cudaSuccess;
}

} // namespace ra
