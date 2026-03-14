#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

Error
PMesh1D::calibrate() {
  ra_invoke(this->calibrate(1));

  Mesh1D mesh_local(this->config.local);
  ra_invoke(local.copy(mesh_local));

  ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  return RA_SUCCESS;
}

} // namespace ra
