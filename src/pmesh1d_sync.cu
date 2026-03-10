#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

Error
PMesh1D::sync() {
  auto rank = this->config.topology.rank;
  for (int d = 0; d < 1; ++d) {
    ra_invoke(local.sync(rank.neighbor[2 * d + 0], d, Direction::Upwind));
    ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

    ra_invoke(local.sync(rank.neighbor[2 * d + 1], d, Direction::Downwind));
    ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));
  }

  return cudaSuccess;
}

} // namespace ra
